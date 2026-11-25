"""
설정 불러오기 — 팀장이 올린 설정 파일을 그대로 읽어서 한 곳에 모읍니다.

  project.json                  Pose 모델 이름, 이미지 크기, 사람·관절 신뢰도 기준
  configs/day07_realtime.json   Window 길이, 예측 간격, 신뢰도 기준 등 (7일차)
  configs/day08_multi_person.json  추적기 설정 (8일차)
  models/rf_training_info.json  학습 당시 정보 (클래스, Sequence 길이 30, Feature 1,140)

왜 숫자를 코드에 직접 쓰지 않고 파일에서 읽나요?
  팀장이 기준값을 바꾸면 json만 고치면 되고, 내 코드는 그대로 둘 수 있기 때문입니다.
  학습 때와 같은 값을 써야 결과가 맞기도 하고요.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
CONFIGS_DIR = BASE_DIR / "configs"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


PROJECT = _load(BASE_DIR / "project.json")
REALTIME = _load(CONFIGS_DIR / "day07_realtime.json")
MULTI = _load(CONFIGS_DIR / "day08_multi_person.json")

# ── 모델 파일 ─────────────────────────────────────
MODEL_FILE = MODELS_DIR / "rf_action_baseline.joblib"
FEATURE_COLUMNS_FILE = MODELS_DIR / "rf_feature_columns.json"
TRAINING_INFO_FILE = MODELS_DIR / "rf_training_info.json"
POSE_WEIGHTS = MODELS_DIR / PROJECT.get("pose_model", "yolo11n-pose.pt")

# ── Pose (project.json) ───────────────────────────
IMAGE_SIZE = int(PROJECT.get("image_size", 640))
PERSON_CONF = float(PROJECT.get("person_confidence", 0.25))
KEYPOINT_CONF = float(PROJECT.get("keypoint_confidence", 0.5))
DEVICE = None if PROJECT.get("device", "auto") == "auto" else PROJECT["device"]

# ── 추적 (day08) ──────────────────────────────────
TRACKER = MULTI.get("tracker", "bytetrack.yaml")

# ── Window·판정 (day07) ───────────────────────────
WINDOW_SECONDS = float(REALTIME.get("window_seconds", 5.0))
WINDOW_STRIDE_SECONDS = float(REALTIME.get("prediction_interval_sec", 0.25))
MIN_WINDOW_FILL_RATIO = float(REALTIME.get("min_window_fill_ratio", 0.8))
MIN_VALID_FEATURE_RATIO = float(REALTIME.get("min_valid_feature_ratio", 0.6))
CONFIDENCE_THRESHOLD = float(REALTIME.get("confidence_threshold", 0.55))

# ── 기타 (환경변수로 변경 가능) ────────────────────
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "1800"))  # 30fps × 60초
DEFAULT_FPS = 30.0  # 영상에 FPS 정보가 없을 때
