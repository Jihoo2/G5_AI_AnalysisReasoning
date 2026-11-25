"""
관절 좌표 → 모델 입력 숫자 1,140개.

rf_feature_columns.json 을 보면 학습 때 입력은 이렇게 생겼습니다.
  한 프레임(Step) = 38개
    - 관절 17개 × (x, y) = 34개   예) t00_nose_x, t00_nose_y, ...
    - 각도 4개                     예) t00_left_elbow_angle, ..., t00_right_knee_angle
  30 Step × 38 = 1,140개           이름 앞의 t00~t29 가 Step 번호

⚠️ 확인 필요 (팀장의 학습 코드 feature_utils.py 와 맞춰야 하는 부분)
  normalize_keypoints() 와 frame_features() 의 '정규화 방법'과 '각도 단위'는
  json 파일에 적혀 있지 않아서 일반적인 방식으로 작성했습니다.
  학습 코드를 받으면 이 두 함수만 그 코드와 똑같이 바꾸면 됩니다.
"""
from __future__ import annotations

import numpy as np

# COCO 17 관절 — YOLO Pose가 돌려주는 순서 = 학습 Feature 이름 순서
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# (Feature 이름, 끝점, 꼭짓점, 끝점)
ANGLES = [
    ("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_knee_angle", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_angle", "right_hip", "right_knee", "right_ankle"),
]

FRAME_FEATURE_NAMES = [f"{k}_{axis}" for k in KEYPOINT_NAMES for axis in ("x", "y")] + [a[0] for a in ANGLES]
FEATURES_PER_STEP = len(FRAME_FEATURE_NAMES)  # 38


def expected_columns(steps: int) -> list[str]:
    """학습 때 Feature 이름 규칙대로 만든 목록 (t00_nose_x ...). 모델 파일과 비교해 검사에 씀."""
    return [f"t{i:02d}_{name}" for i in range(steps) for name in FRAME_FEATURE_NAMES]


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def normalize_keypoints(xy: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """
    ⚠️ 학습 코드와 맞춰야 하는 함수.
    현재 방식: 엉덩이 중심을 (0,0)으로 옮기고, 어깨 중심~엉덩이 중심 거리(몸통 길이)로 나눔.
    왜? 사람이 화면 어디에 있든, 카메라와 얼마나 떨어져 있든 같은 자세면 같은 숫자가 나오게 하려고.
    """
    hip = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) / 2
    sh = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) / 2
    hips_ok = visible[KP["left_hip"]] and visible[KP["right_hip"]]
    center = hip if hips_ok else sh
    scale = float(np.linalg.norm(sh - hip))
    if not (visible[[KP["left_hip"], KP["right_hip"], KP["left_shoulder"], KP["right_shoulder"]]].all()) or scale < 1e-3:
        pts = xy[visible] if visible.any() else xy
        scale = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0))
    return (xy - center) / scale


def frame_features(xy: np.ndarray, conf: np.ndarray, keypoint_conf: float) -> np.ndarray:
    """
    한 프레임 → 38개 숫자. 신뢰도가 keypoint_conf 미만인 관절은 NaN(빈 값).
    ⚠️ 각도 단위는 '도(0~180)'로 가정했습니다. 학습 코드가 라디안이나 0~1이면 바꿔야 합니다.
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(17, 2)
    visible = np.asarray(conf, dtype=np.float32) >= keypoint_conf
    norm = normalize_keypoints(xy, visible)
    norm[~visible] = np.nan
    angles = [
        _angle(xy[KP[a]], xy[KP[b]], xy[KP[c]]) if visible[[KP[a], KP[b], KP[c]]].all() else np.nan
        for _, a, b, c in ANGLES
    ]
    return np.concatenate([norm.reshape(-1), np.asarray(angles, dtype=np.float32)]).astype(np.float32)


def to_sequence(frames: np.ndarray, steps: int) -> np.ndarray:
    """
    길이가 제각각인 프레임 묶음 (T, 38) → 정확히 steps(30)개로 맞춤.
    Feature마다 빈 값(NaN)을 뺀 나머지로 시간축 선형 보간합니다.
    왜? 동작을 빨리 하든 천천히 하든 모델은 항상 30칸만 받기 때문입니다.
    """
    frames = np.asarray(frames, dtype=np.float32)
    t = len(frames)
    src = np.linspace(0, 1, t)
    dst = np.linspace(0, 1, steps)
    out = np.full((steps, frames.shape[1]), np.nan, dtype=np.float32)
    for j in range(frames.shape[1]):
        col = frames[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            out[:, j] = np.interp(dst, src[ok], col[ok])
        elif ok.sum() == 1:
            out[:, j] = col[ok][0]
    return out


def flatten(seq: np.ndarray, columns: list[str]) -> np.ndarray:
    """(30, 38) → 학습 때 순서(columns) 그대로 1,140개. 남은 빈 값은 0으로 채움."""
    values = {f"t{i:02d}_{name}": seq[i, j] for i in range(seq.shape[0]) for j, name in enumerate(FRAME_FEATURE_NAMES)}
    vec = np.asarray([values[c] for c in columns], dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0)
