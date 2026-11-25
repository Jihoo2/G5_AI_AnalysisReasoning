"""
YOLO·영상 없이 확인할 수 있는 부분을 테스트합니다.
실행 (ai_server 폴더):  python -m pytest tests -q

1) 내 Feature 이름 규칙이 팀장 rf_feature_columns.json (1,140개)과 순서까지 똑같은가
2) 30 Step 변환, Window 나누기가 의도대로 되는가
3) 가짜 Random Forest로 추론 → OK/NG 판정까지 이어지는가
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_config as cfg  # noqa: E402
from services import model_loader  # noqa: E402
from services.features import FEATURES_PER_STEP, expected_columns, frame_features, to_sequence  # noqa: E402
from services.inference import judge, make_windows, predict_track  # noqa: E402
from services.pose_pipeline import Track  # noqa: E402

CLASSES = ["bend_return", "drink_return", "leg_raise_lower", "sit_stand", "walk_turn_walk", "wave"]


def _person(t: float) -> tuple[np.ndarray, np.ndarray]:
    """손을 흔드는 듯 움직이는 가짜 사람 1프레임 (관절 17개, 신뢰도 0.9)."""
    xy = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 150], [360, 150], [260, 210], [380, 210],
        [250, 270], [400 + 30 * np.sin(t), 150 + 30 * np.cos(t)],
        [290, 280], [350, 280], [290, 370], [350, 370], [290, 460], [350, 460],
    ], dtype=np.float32)
    return xy, np.full(17, 0.9, dtype=np.float32)


def test_feature_names_match_team_file():
    team = json.loads(cfg.FEATURE_COLUMNS_FILE.read_text(encoding="utf-8"))
    info = json.loads(cfg.TRAINING_INFO_FILE.read_text(encoding="utf-8"))
    assert FEATURES_PER_STEP == 38
    assert len(team) == info["feature_count"] == 1140
    assert expected_columns(info["sequence_steps"]) == team


def test_low_confidence_keypoint_becomes_nan():
    xy, conf = _person(0)
    conf[0] = 0.1  # 코가 잘 안 보임
    f = frame_features(xy, conf, cfg.KEYPOINT_CONF)
    assert f.shape == (38,)
    assert np.isnan(f[0]) and np.isnan(f[1])
    assert not np.isnan(f[36])  # 왼무릎 각도는 정상


def test_sequence_is_30_steps():
    frames = np.random.default_rng(0).normal(size=(87, 38)).astype(np.float32)
    assert to_sequence(frames, 30).shape == (30, 38)


def test_windows_5s_with_quarter_second_stride():
    frames = list(range(300))  # 30fps × 10초
    w = make_windows(frames, 30.0, 10)
    # 5초 = 150프레임, 0.25초 ≈ 8프레임 간격 → 시작 0,8,...,144 (19개) + 끝 맞춤 150 = 20개
    assert len(w) == 20
    assert all(len(x) == 150 for x in w)
    assert w[-1][-1] == 299  # 영상 마지막 프레임까지 포함


def test_short_clip_uses_one_window():
    assert len(make_windows(list(range(60)), 30.0, 10)) == 1
    assert make_windows(list(range(5)), 30.0, 10) == []


@pytest.fixture()
def dummy_model(tmp_path, monkeypatch):
    team_cols = json.loads(cfg.FEATURE_COLUMNS_FILE.read_text(encoding="utf-8"))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 1140))
    y = np.repeat(CLASSES, 20)
    X[y == "wave", 1] += 3  # wave 는 특정 값이 크게 → 구분 가능하게
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    path = tmp_path / "rf.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(cfg, "MODEL_FILE", path)
    lm = model_loader.load()
    assert lm.columns == team_cols
    return lm


def test_predict_and_judge(dummy_model):
    track = Track(1)
    for i in range(240):  # 8초 @30fps
        xy, conf = _person(i / 5)
        track.frame_idx.append(i)
        track.features.append(frame_features(xy, conf, cfg.KEYPOINT_CONF))
    pred = predict_track(track, 30.0, dummy_model)
    assert pred is not None
    assert pred["action_label"] in CLASSES
    assert pred["window_count"] >= 1
    assert abs(sum(pred["probabilities"].values()) - 1) < 0.01

    assert judge(pred, None)["verdict"] is None
    other = next(c for c in CLASSES if c != pred["action_label"])
    assert judge(pred, other)["verdict"] == "NG"
    sure = {**pred, "is_uncertain": False, "confidence": 0.9}
    assert judge(sure, pred["action_label"])["verdict"] == "OK"
