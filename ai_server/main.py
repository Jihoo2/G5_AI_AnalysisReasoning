"""
G5 AI 서버 (FastAPI) — 이 파일 하나로 동작합니다.

  실행: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
  문서: http://localhost:8001/docs

하는 일: 영상 → 관절(YOLO Pose) → 숫자 1,140개 → Random Forest → 인식 동작
         파일 이름의 정답(S01_A001_T01.mp4 → A001)과 같으면 "OK", 다르면 "NG" 만 응답
하지 않는 일: DB 저장·회원·화면 (Django backend/ 담당)

파일 구성 (위에서 아래로 읽으면 처리 순서와 같습니다)
  1. 설정       팀장 파일(project.json, configs/, models/)에서 값 읽기
  2. Feature    관절 좌표 → 프레임당 38개 → 30 Step → 1,140개
  3. 모델 로드   Random Forest + json, 서버 시작 때 1번만
  4. Pose       영상 → YOLO Pose + 추적 → 사람별 Feature
  5. 판정       5초 Window 예측·평균 → 동작 → OK/NG → summary
  6. API        /health, /actions, /predict/video (업로드), /data, /predict/file/{파일명}
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse


# ════════════════════════════════════════════════════════════
# 1. 설정 — 팀장 파일에서 읽기
#    왜 숫자를 코드에 직접 쓰지 않나요? 학습 때와 같은 값을 써야 결과가 맞고,
#    팀장이 json만 고치면 내 코드는 그대로 둘 수 있기 때문입니다.
# ════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))  # 테스트 영상 보관 폴더


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


PROJECT = _load_json(BASE_DIR / "project.json")                       # Pose 설정
REALTIME = _load_json(BASE_DIR / "configs" / "day07_realtime.json")    # Window·신뢰도
MULTI = _load_json(BASE_DIR / "configs" / "day08_multi_person.json")   # 추적기

MODEL_FILE = MODELS_DIR / "rf_action_baseline.joblib"
FEATURE_COLUMNS_FILE = MODELS_DIR / "rf_feature_columns.json"
TRAINING_INFO_FILE = MODELS_DIR / "rf_training_info.json"
POSE_WEIGHTS = MODELS_DIR / PROJECT.get("pose_model", "yolo11n-pose.pt")

IMAGE_SIZE = int(PROJECT.get("image_size", 640))
PERSON_CONF = float(PROJECT.get("person_confidence", 0.25))
KEYPOINT_CONF = float(PROJECT.get("keypoint_confidence", 0.5))
DEVICE = None if PROJECT.get("device", "auto") == "auto" else PROJECT["device"]
TRACKER = MULTI.get("tracker", "bytetrack.yaml")

WINDOW_SECONDS = float(REALTIME.get("window_seconds", 5.0))              # 5초씩 묶어 판단
WINDOW_STRIDE_SECONDS = float(REALTIME.get("prediction_interval_sec", 0.25))  # 0.25초씩 밀면서
MIN_WINDOW_FILL_RATIO = float(REALTIME.get("min_window_fill_ratio", 0.8))     # 구간의 80% 이상 보여야
MIN_VALID_FEATURE_RATIO = float(REALTIME.get("min_valid_feature_ratio", 0.6))  # 숫자의 60% 이상 유효해야
CONFIDENCE_THRESHOLD = float(REALTIME.get("confidence_threshold", 0.55))      # 이보다 낮으면 '불확실'

# 프레임 N장에 1장만 분석 (2 = 약 2배 빠름, 1 = 전부). 5초에 75장이라 30 Step 만들기엔 충분.
FRAME_STRIDE = max(1, int(os.getenv("FRAME_STRIDE", "2")))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "1800"))
DEFAULT_FPS = 30.0
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# 파일 이름의 동작 ID → 모델 라벨.  예) S01_A001_T01.mp4 → A001 → bend_return (정답)
ACTION_IDS = {
    "A001": "bend_return",
    "A002": "leg_raise_lower",
    "A003": "walk_turn_walk",
    "A004": "drink_return",
    "A005": "sit_stand",
    "A006": "wave",
}

# 화면 표시용 한글 이름 — 백엔드·화면이 변환표를 따로 만들 필요 없게 응답에 함께 넣음
ACTION_NAMES_KO = {
    "bend_return": "허리 굽혔다 펴기",
    "drink_return": "음료 마시기",
    "leg_raise_lower": "한쪽 다리 들었다 내리기",
    "sit_stand": "앉았다 일어나기",
    "walk_turn_walk": "걷다가 방향 바꾸기",
    "wave": "손 흔들기",
}


# ════════════════════════════════════════════════════════════
# 2. Feature — 관절 좌표 → 모델 입력 1,140개
#    rf_feature_columns.json 규칙: 한 프레임 = 관절 17개 × (x, y) 34개 + 각도 4개 = 38개
#                                 30 Step × 38 = 1,140개, 이름은 t00_nose_x ... t29_right_knee_angle
#
#  ⚠️ 확인 필요: normalize_keypoints()의 정규화 방법과 각도 단위(도)는 json에 없어서 가정했습니다.
#     팀장의 학습 코드 feature_utils.py를 받으면 이 부분만 똑같이 맞추면 됩니다.
# ════════════════════════════════════════════════════════════
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
ANGLES = [  # (Feature 이름, 끝점, 꼭짓점, 끝점)
    ("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_knee_angle", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_angle", "right_hip", "right_knee", "right_ankle"),
]
FRAME_FEATURE_NAMES = [f"{k}_{a}" for k in KEYPOINT_NAMES for a in ("x", "y")] + [a[0] for a in ANGLES]


def expected_columns(steps: int) -> list[str]:
    """학습 Feature 이름 규칙대로 만든 목록. 모델의 json과 같은지 검사할 때 사용."""
    return [f"t{i:02d}_{name}" for i in range(steps) for name in FRAME_FEATURE_NAMES]


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def normalize_keypoints(xy: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """⚠️ 학습 코드와 맞춰야 하는 함수.
    엉덩이 중심을 (0,0)으로, 몸통 길이(어깨 중심~엉덩이 중심)로 나눔
    → 화면 위치·카메라 거리가 달라도 같은 자세면 같은 숫자가 나오게."""
    hip = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) / 2
    sh = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) / 2
    center = hip if visible[KP["left_hip"]] and visible[KP["right_hip"]] else sh
    scale = float(np.linalg.norm(sh - hip))
    torso = [KP["left_hip"], KP["right_hip"], KP["left_shoulder"], KP["right_shoulder"]]
    if not visible[torso].all() or scale < 1e-3:
        pts = xy[visible] if visible.any() else xy
        scale = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0))
    return (xy - center) / scale


def frame_features(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """한 프레임 → 38개. 신뢰도가 낮은 관절은 NaN(빈 값)."""
    xy = np.asarray(xy, dtype=np.float32).reshape(17, 2)
    visible = np.asarray(conf, dtype=np.float32) >= KEYPOINT_CONF
    norm = normalize_keypoints(xy, visible)
    norm[~visible] = np.nan
    angles = [_angle(xy[KP[a]], xy[KP[b]], xy[KP[c]]) if visible[[KP[a], KP[b], KP[c]]].all() else np.nan
              for _, a, b, c in ANGLES]
    return np.concatenate([norm.reshape(-1), np.asarray(angles, dtype=np.float32)]).astype(np.float32)


def to_sequence(frames: np.ndarray, steps: int) -> np.ndarray:
    """(T, 38) → (steps, 38). 동작을 빨리 하든 천천히 하든 모델은 항상 30칸만 받기 때문."""
    frames = np.asarray(frames, dtype=np.float32)
    src, dst = np.linspace(0, 1, len(frames)), np.linspace(0, 1, steps)
    out = np.full((steps, frames.shape[1]), np.nan, dtype=np.float32)
    for j in range(frames.shape[1]):
        ok = ~np.isnan(frames[:, j])
        if ok.sum() >= 2:
            out[:, j] = np.interp(dst, src[ok], frames[ok, j])
        elif ok.sum() == 1:
            out[:, j] = frames[ok, j][0]
    return out


def flatten(seq: np.ndarray, columns: list[str]) -> np.ndarray:
    """(30, 38) → 학습 순서(columns) 그대로 1,140개. 빈 값은 0."""
    values = {f"t{i:02d}_{n}": seq[i, j] for i in range(seq.shape[0]) for j, n in enumerate(FRAME_FEATURE_NAMES)}
    return np.nan_to_num(np.asarray([values[c] for c in columns], dtype=np.float32), nan=0.0)


# ════════════════════════════════════════════════════════════
# 3. 모델 로드 — 서버 시작 때 1번만 (요청마다 읽으면 매번 몇 초씩 느려짐)
# ════════════════════════════════════════════════════════════
@dataclass
class LoadedModel:
    model: object
    columns: list[str]
    classes: list[str]
    steps: int
    min_valid_frames: int
    warnings: list[str] = field(default_factory=list)


STATE: dict = {"rf": None, "rf_error": None, "pose": None, "pose_error": None}


def load_model() -> LoadedModel:
    for p in (MODEL_FILE, FEATURE_COLUMNS_FILE, TRAINING_INFO_FILE):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} 이 models/ 폴더에 없습니다.")
    model = joblib.load(MODEL_FILE)  # 팀에서 만든 믿을 수 있는 파일에만 사용
    columns = json.loads(FEATURE_COLUMNS_FILE.read_text(encoding="utf-8"))
    info = json.loads(TRAINING_INFO_FILE.read_text(encoding="utf-8"))
    steps = int(info.get("sequence_steps", 30))
    classes = [str(c) for c in getattr(model, "classes_", info.get("classes", []))]

    if getattr(model, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("모델 입력 수와 rf_feature_columns.json 길이가 다릅니다.")
    if columns != expected_columns(steps):
        raise ValueError("rf_feature_columns.json 의 이름·순서가 main.py의 Feature 규칙과 다릅니다.")
    warnings = []
    if info.get("classes") and set(map(str, info["classes"])) != set(classes):
        warnings.append("rf_training_info.json 의 classes 와 모델의 classes_ 가 다릅니다.")
    return LoadedModel(model, columns, classes, steps, int(info.get("min_valid_frames", 10)), warnings)


# ════════════════════════════════════════════════════════════
# 4. Pose — 영상 → YOLO Pose + 추적 → 사람(Track ID)별 프레임 Feature
# ════════════════════════════════════════════════════════════
class VideoReadError(Exception):
    """영상을 열 수 없거나 손상된 경우."""


@dataclass
class Track:
    track_id: int
    frame_idx: list[int] = field(default_factory=list)      # 이 사람이 보인 (분석한) 프레임 번호
    features: list[np.ndarray] = field(default_factory=list)  # 프레임마다 38개


_pose_lock = threading.Lock()  # YOLO는 한 번에 한 요청만 (동시에 쓰면 추적 번호가 섞임)


def get_pose_model():
    if STATE["pose"] is None:
        from ultralytics import YOLO
        STATE["pose"] = YOLO(str(POSE_WEIGHTS if POSE_WEIGHTS.exists() else POSE_WEIGHTS.name))
    return STATE["pose"]


def inspect_video(path: str) -> tuple[float, int, int, int]:
    """영상이 읽히는지 먼저 확인 → '손상 파일'과 '사람 없음'을 구분하기 위해."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoReadError("영상 파일을 열 수 없습니다. MP4 등 지원 형식인지 확인하세요.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoReadError("영상에서 프레임을 읽을 수 없습니다. 파일이 손상되었을 수 있습니다.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or DEFAULT_FPS
        return fps, frame.shape[1], frame.shape[0], int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def extract_tracks(path: str) -> tuple[dict[int, Track], int]:
    """반환: (사람별 Track, 분석한 프레임 수)"""
    tracks: dict[int, Track] = {}
    processed = 0
    with _pose_lock:
        results = get_pose_model().track(
            source=path, stream=True, persist=False, verbose=False,   # persist=False: 영상마다 ID 1부터
            conf=PERSON_CONF, imgsz=IMAGE_SIZE, tracker=TRACKER, device=DEVICE,
            vid_stride=FRAME_STRIDE,                                   # N장에 1장만 분석
        )
        for i, r in enumerate(results):
            processed += 1
            if processed > MAX_FRAMES:
                break
            if r.boxes is None or r.boxes.id is None or r.keypoints is None:
                continue
            ids = r.boxes.id.int().cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()
            conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
            for k, tid in enumerate(ids):
                t = tracks.setdefault(int(tid), Track(int(tid)))
                t.frame_idx.append(i)
                t.features.append(frame_features(xy[k], conf[k]))
    return tracks, min(processed, MAX_FRAMES)


# ════════════════════════════════════════════════════════════
# 5. 판정 — 5초 Window를 0.25초씩 밀며 예측 → 확률 평균 → 동작 → OK/NG → summary
#    왜 평균? 한 번만 보면 순간적인 흔들림에 결과가 바뀜 (7일차 '예측 안정화'와 같은 목적)
# ════════════════════════════════════════════════════════════
def make_windows(frame_idx: list[int], fps: float, min_valid_frames: int) -> list[np.ndarray]:
    """프레임 번호 목록 → Window마다 사용할 위치 목록."""
    idx = np.asarray(frame_idx)
    win = max(1, round(WINDOW_SECONDS * fps))
    stride = max(1, round(WINDOW_STRIDE_SECONDS * fps))
    first, last = int(idx[0]), int(idx[-1])
    if last - first + 1 < win:  # 5초보다 짧으면 전체를 Window 1개로
        return [np.arange(len(idx))] if len(idx) >= min_valid_frames else []
    starts = list(range(first, last - win + 2, stride))
    if starts[-1] + win - 1 < last:  # 영상 끝부분도 빠지지 않게
        starts.append(last - win + 1)
    windows = []
    for s in starts:
        sel = np.where((idx >= s) & (idx < s + win))[0]
        if len(sel) >= min_valid_frames and len(sel) / win >= MIN_WINDOW_FILL_RATIO:
            windows.append(sel)
    return windows


def predict_track(track: Track, fps: float, lm: LoadedModel) -> dict | None:
    feats = np.stack(track.features)
    X = [flatten(to_sequence(feats[sel], lm.steps), lm.columns)
         for sel in make_windows(track.frame_idx, fps, lm.min_valid_frames)
         if np.mean(~np.isnan(feats[sel])) >= MIN_VALID_FEATURE_RATIO]  # 너무 가려진 구간은 제외
    if not X:
        return None
    proba = lm.model.predict_proba(np.stack(X))
    mean = proba.mean(axis=0)
    best = int(np.argmax(mean))
    label = lm.classes[best]
    confident = proba.max(axis=1) >= CONFIDENCE_THRESHOLD
    votes = np.bincount(proba.argmax(axis=1)[confident], minlength=len(lm.classes))
    return {
        "action_label": label,
        "action_name": ACTION_NAMES_KO.get(label, label),
        "confidence": round(float(mean[best]), 4),
        "is_uncertain": bool(mean[best] < CONFIDENCE_THRESHOLD),
        "window_count": len(X),
        "frame_count": len(track.frame_idx),
        "votes": {c: int(v) for c, v in zip(lm.classes, votes) if v},
        "probabilities": {c: round(float(p), 4) for c, p in zip(lm.classes, mean)},
    }


def label_from_filename(filename: str) -> str | None:
    """파일 이름에서 정답 동작 찾기.  S01_A001_T01.mp4 → 'A001' → 'bend_return'.  없으면 None."""
    m = re.search(r"A\d{3}", Path(filename).name.upper())
    return ACTION_IDS.get(m.group(0)) if m else None


def judge(pred: dict, expected_action: str | None) -> dict:
    """OK/NG 규칙: AI가 인식한 동작 == 정답 동작 → OK, 다르면 NG."""
    if not expected_action:
        return {"verdict": None, "reason": "정답 동작을 알 수 없습니다 (파일 이름에 A001~A006 없음)."}
    want = ACTION_NAMES_KO.get(expected_action, expected_action)
    if pred["action_label"] != expected_action:
        return {"verdict": "NG", "reason": f"정답 '{want}' ≠ 인식 '{pred['action_name']}'"}
    return {"verdict": "OK", "reason": f"정답 '{want}' = 인식 '{pred['action_name']}' (신뢰도 {pred['confidence']:.0%})"}


def build_summary(results: list[dict], message: str) -> dict:
    """백엔드가 그대로 저장·표시할 '대표 1명' 결과 (가장 오래 보인 사람).
    results는 여러 명 목록이라 반복문이 필요하지만, summary는 값을 바로 쓰면 됨."""
    if not results:
        return {"track_id": None, "people_count": 0, "action_label": None, "action_name": None,
                "confidence": None, "confidence_percent": None, "is_uncertain": None,
                "verdict": None, "display_text": message}
    main = max(results, key=lambda r: r["frame_count"])
    pct = round(main["confidence"] * 100)
    text = f"{main['action_name']} ({pct}%)" + (" · 판단 불확실" if main["is_uncertain"] else "")
    if main["verdict"]:
        text = f"{main['verdict']} · {text} · {main['reason']}"
    return {"track_id": main["track_id"], "people_count": len(results),
            "action_label": main["action_label"], "action_name": main["action_name"],
            "confidence": main["confidence"], "confidence_percent": pct,
            "is_uncertain": main["is_uncertain"], "verdict": main["verdict"], "display_text": text}


# ════════════════════════════════════════════════════════════
# 6. API
# ════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 모델 2개를 미리 불러둠 → 첫 요청이 느려 Celery가 타임아웃으로 오해하는 것 방지
    try:
        STATE["rf"] = load_model()
    except Exception as exc:  # noqa: BLE001 — 실패해도 서버는 켜고 /health 에서 원인 표시
        STATE["rf_error"] = f"{exc.__class__.__name__}: {exc}"
    try:
        get_pose_model().predict(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                                 imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)  # warm-up
    except Exception as exc:  # noqa: BLE001
        STATE["pose"], STATE["pose_error"] = None, f"{exc.__class__.__name__}: {exc}"
    yield


app = FastAPI(title="G5 AI 서버", version="0.4.0", lifespan=lifespan)


def _error(status: int, code: str, message: str) -> HTTPException:
    """오류 모양 통일: code는 프로그램이 비교하는 값, message는 사람이 읽는 문장."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@app.get("/health", tags=["health"])
def health():
    """서버·모델 준비 상태 (Docker healthcheck에도 사용)."""
    lm: LoadedModel | None = STATE["rf"]
    ready = lm is not None and STATE["pose"] is not None
    return {
        "status": "ok" if ready else "error",
        "rf_model_ready": lm is not None,
        "pose_model_ready": STATE["pose"] is not None,
        "pose_weights": POSE_WEIGHTS.name,
        "frame_stride": FRAME_STRIDE,
        "classes": lm.classes if lm else [],
        "feature_count": len(lm.columns) if lm else 0,
        "sequence_steps": lm.steps if lm else None,
        "warnings": lm.warnings if lm else [],
        "errors": [e for e in (STATE["rf_error"], STATE["pose_error"]) if e],
    }


@app.get("/actions", tags=["predictions"])
def actions():
    """인식할 수 있는 동작과 판정 기준 (팀원 확인용)."""
    lm: LoadedModel | None = STATE["rf"]
    return {
        "actions": [{"label": c, "name": ACTION_NAMES_KO.get(c, c)} for c in (lm.classes if lm else [])],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "window_seconds": WINDOW_SECONDS,
        "frame_stride": FRAME_STRIDE,
    }


def analyze_file(path: str, expected: str | None) -> dict:
    """영상 파일 1개 분석 → 상세 결과(JSON). 업로드·data 폴더 두 API가 함께 사용."""
    lm: LoadedModel = STATE["rf"]
    started = time.perf_counter()
    try:
        fps, width, height, frame_count = inspect_video(path)
    except VideoReadError as exc:
        raise _error(400, "VIDEO_UNREADABLE", str(exc)) from exc
    tracks, processed = extract_tracks(path)

    results = []
    for tid in sorted(tracks):
        pred = predict_track(tracks[tid], fps / FRAME_STRIDE, lm)
        if pred:
            results.append({"track_id": tid, **pred, **judge(pred, expected)})

    if results:
        status, message = "OK", f"{len(results)}명의 동작을 분석했습니다."
    elif tracks:
        status, message = "TOO_SHORT", "사람은 찾았지만 판단할 만큼 오래·선명하게 보이지 않았습니다."
    else:
        status, message = "NO_PERSON", "영상에서 사람을 찾지 못했습니다."
    summary = build_summary(results, message)

    return {
        "verdict": summary["verdict"] or "NG",  # 사람을 못 찾는 등 판단 불가도 NG
        "status": status,
        "message": message,
        "expected_action": expected,
        "summary": summary,
        "results": results,
        "video": {"fps": fps, "width": width, "height": height, "frame_count": frame_count},
        "frames_processed": processed,
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
    }


def _respond(result: dict, detail: bool):
    """기본은 'OK' 또는 'NG' 글자만. detail=true 면 상세 JSON (확인·백엔드용)."""
    return result if detail else PlainTextResponse(result["verdict"])


def _check_ready() -> LoadedModel:
    lm: LoadedModel | None = STATE["rf"]
    if lm is None or STATE["pose"] is None:
        raise _error(503, "MODEL_NOT_READY", STATE["rf_error"] or STATE["pose_error"] or "모델이 준비되지 않았습니다.")
    return lm


def _expected(filename: str, expected_action: str | None, lm: LoadedModel) -> str:
    """정답 동작: expected_action 이 있으면 그것, 없으면 파일 이름(A001~A006)에서."""
    value = (expected_action or "").strip()
    if value and value != "string":  # 'string' = /docs 화면의 예시값
        if value not in lm.classes:
            raise _error(422, "UNKNOWN_ACTION", f"모르는 동작입니다: {value} (가능: {', '.join(lm.classes)})")
        return value
    label = label_from_filename(filename)
    if label is None:
        raise _error(422, "NO_LABEL_IN_FILENAME", f"파일 이름에서 정답 동작(A001~A006)을 찾을 수 없습니다: {filename}")
    return label


@app.post("/predict/video", tags=["predictions"])
def predict_video(
    file: UploadFile = File(..., description="영상 (예: S01_A001_T01.mp4 — 이름의 A001이 정답)"),
    expected_action: str | None = Form(None, description="정답 동작을 직접 줄 때만 (비우면 파일 이름에서 읽음)"),
    detail: bool = Query(False, description="true면 상세 JSON"),
):
    """업로드한 영상 → 'OK'(인식 = 정답) 또는 'NG'(다름)."""
    lm = _check_ready()
    name = file.filename or ""
    expected = _expected(name, expected_action, lm)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise _error(415, "UNSUPPORTED_FORMAT", f"지원하지 않는 형식입니다: {ext or '(확장자 없음)'}")

    fd, tmp = tempfile.mkstemp(suffix=ext)  # OpenCV·YOLO는 파일 경로로 읽으므로 임시 저장
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)
        size_mb = os.path.getsize(tmp) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise _error(413, "FILE_TOO_LARGE", f"{size_mb:.1f}MB — 최대 {MAX_UPLOAD_MB}MB")
        return _respond(analyze_file(tmp, expected), detail)
    finally:
        file.file.close()
        Path(tmp).unlink(missing_ok=True)


@app.get("/data", tags=["data"])
def list_data():
    """data/ 폴더에 있는 영상 목록."""
    files = sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in ALLOWED_EXT) if DATA_DIR.exists() else []
    return {"data_dir": str(DATA_DIR), "files": files}


@app.get("/predict/file/{filename}", tags=["data"])
def predict_data_file(filename: str, detail: bool = Query(False, description="true면 상세 JSON")):
    """data/ 폴더의 영상을 이름으로 분석 → 'OK' 또는 'NG'.  예) /predict/file/S01_A001_T01.mp4"""
    lm = _check_ready()
    path = DATA_DIR / Path(filename).name  # 폴더 밖 파일 접근 방지
    if not path.exists():
        raise _error(404, "FILE_NOT_FOUND", f"data 폴더에 없는 파일입니다: {path.name}")
    if path.suffix.lower() not in ALLOWED_EXT:
        raise _error(415, "UNSUPPORTED_FORMAT", f"지원하지 않는 형식입니다: {path.suffix}")
    return _respond(analyze_file(str(path), _expected(path.name, None, lm)), detail)
