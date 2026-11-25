"""
팀장 모델(joblib) + json 2개를 서버 시작 때 한 번만 불러와 재사용합니다.
불러오면서 '학습 때와 같은 입력을 만들 수 있는지' 검사합니다.

왜 시작 때 한 번만?
  요청마다 모델을 새로 읽으면 매번 몇 초씩 느려지기 때문입니다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib

import app_config as cfg
from services.features import expected_columns


@dataclass
class LoadedModel:
    model: object
    columns: list[str]
    classes: list[str]
    steps: int
    min_valid_frames: int
    warnings: list[str] = field(default_factory=list)


_state: dict = {"model": None, "error": None}


def load() -> LoadedModel:
    for p in (cfg.MODEL_FILE, cfg.FEATURE_COLUMNS_FILE, cfg.TRAINING_INFO_FILE):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} 이 models/ 폴더에 없습니다.")

    model = joblib.load(cfg.MODEL_FILE)  # 팀에서 만든 믿을 수 있는 파일에만 사용
    columns = json.loads(cfg.FEATURE_COLUMNS_FILE.read_text(encoding="utf-8"))
    info = json.loads(cfg.TRAINING_INFO_FILE.read_text(encoding="utf-8"))
    steps = int(info.get("sequence_steps", 30))
    classes = [str(c) for c in getattr(model, "classes_", info.get("classes", []))]
    warnings: list[str] = []

    n_in = getattr(model, "n_features_in_", len(columns))
    if n_in != len(columns):
        raise ValueError(f"모델 입력 수({n_in})와 Feature 목록({len(columns)})이 다릅니다.")
    if columns != expected_columns(steps):
        raise ValueError("rf_feature_columns.json 의 이름·순서가 features.py 규칙과 다릅니다. features.py를 확인하세요.")
    if info.get("classes") and set(map(str, info["classes"])) != set(classes):
        warnings.append("rf_training_info.json 의 classes 와 모델의 classes_ 가 다릅니다.")

    return LoadedModel(model, columns, classes, steps, int(info.get("min_valid_frames", 10)), warnings)


def init() -> None:
    try:
        _state["model"], _state["error"] = load(), None
    except Exception as exc:  # noqa: BLE001 — 실패해도 서버는 켜고 /health 에서 원인 표시
        _state["model"], _state["error"] = None, f"{exc.__class__.__name__}: {exc}"


def get() -> LoadedModel | None:
    return _state["model"]


def error() -> str | None:
    return _state["error"]
