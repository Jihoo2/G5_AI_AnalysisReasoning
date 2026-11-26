"""
YOLO·실제 영상 없이 확인할 수 있는 부분 테스트.  실행 (ai_server 폴더):  python -m pytest tests -q

1) Feature 이름 1,140개가 팀장 rf_feature_columns.json 과 순서까지 같은가
2) 30 Step 변환, Window 나누기
3) 가짜 Random Forest로 추론 → OK/NG → summary
4) 백엔드용 client/ai_client.py 의 성공·실패 처리 (가짜 응답 사용)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import requests
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client"))

import ai_client  # noqa: E402
import main  # noqa: E402

CLASSES = ["bend_return", "drink_return", "leg_raise_lower", "sit_stand", "walk_turn_walk", "wave"]


def _person(t: float):
    """손을 흔드는 듯 움직이는 가짜 사람 1프레임 (관절 17개, 신뢰도 0.9)."""
    xy = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 150], [360, 150], [260, 210], [380, 210],
        [250, 270], [400 + 30 * np.sin(t), 150 + 30 * np.cos(t)],
        [290, 280], [350, 280], [290, 370], [350, 370], [290, 460], [350, 460],
    ], dtype=np.float32)
    return xy, np.full(17, 0.9, dtype=np.float32)


# ── 1·2. Feature ────────────────────────────────────
def test_feature_names_match_team_file():
    team = json.loads(main.FEATURE_COLUMNS_FILE.read_text(encoding="utf-8"))
    info = json.loads(main.TRAINING_INFO_FILE.read_text(encoding="utf-8"))
    assert len(main.FRAME_FEATURE_NAMES) == 38
    assert len(team) == info["feature_count"] == 1140
    assert main.expected_columns(info["sequence_steps"]) == team


def test_low_confidence_keypoint_becomes_nan():
    xy, conf = _person(0)
    conf[0] = 0.1
    f = main.frame_features(xy, conf)
    assert f.shape == (38,) and np.isnan(f[0]) and not np.isnan(f[36])


def test_sequence_is_30_steps():
    assert main.to_sequence(np.random.default_rng(0).normal(size=(87, 38)), 30).shape == (30, 38)


def test_windows_cover_whole_video():
    w = main.make_windows(list(range(300)), 30.0, 10)  # 30fps × 10초
    assert len(w) == 20 and all(len(x) == 150 for x in w) and w[-1][-1] == 299


def test_short_clip_uses_one_window():
    assert len(main.make_windows(list(range(60)), 30.0, 10)) == 1
    assert main.make_windows(list(range(5)), 30.0, 10) == []


# ── 3. 추론·판정·summary ─────────────────────────────
@pytest.fixture()
def dummy_model(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(120, 1140)), np.repeat(CLASSES, 20)
    path = tmp_path / "rf.joblib"
    joblib.dump(RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y), path)
    monkeypatch.setattr(main, "MODEL_FILE", path)
    return main.load_model()


def test_predict_judge_summary(dummy_model):
    track = main.Track(1)
    for i in range(120):  # 8초 영상을 2장에 1장 분석 = 15fps × 8초
        xy, conf = _person(i / 3)
        track.frame_idx.append(i)
        track.features.append(main.frame_features(xy, conf))
    pred = main.predict_track(track, 15.0, dummy_model)
    assert pred and pred["action_label"] in CLASSES and pred["action_name"]
    assert abs(sum(pred["probabilities"].values()) - 1) < 0.01

    assert main.judge(pred, None)["verdict"] is None
    other = next(c for c in CLASSES if c != pred["action_label"])
    assert main.judge(pred, other)["verdict"] == "NG"
    sure = {**pred, "is_uncertain": False, "confidence": 0.9}
    assert main.judge(sure, pred["action_label"])["verdict"] == "OK"

    s = main.build_summary([{"track_id": 1, **sure, **main.judge(sure, sure["action_label"])}], "1명")
    assert s["verdict"] == "OK" and s["confidence_percent"] == 90 and s["display_text"].startswith("OK · ")


def test_summary_picks_longest_person_and_empty():
    a = {"track_id": 1, "action_label": "wave", "action_name": "손 흔들기", "confidence": 0.8,
         "is_uncertain": False, "frame_count": 40, "verdict": None, "reason": ""}
    b = {**a, "track_id": 2, "frame_count": 150}
    assert main.build_summary([a, b], "")["track_id"] == 2
    empty = main.build_summary([], "영상에서 사람을 찾지 못했습니다.")
    assert empty["people_count"] == 0 and empty["display_text"] == "영상에서 사람을 찾지 못했습니다."


# ── 4. 백엔드용 ai_client ────────────────────────────
class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


@pytest.fixture()
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake")
    return p


def test_client_success(monkeypatch, video):
    body = {"summary": {"verdict": "OK"}, "processing_time_ms": 1000,
            "results": [{"track_id": 1, "action_label": "wave", "confidence": 0.8, "probabilities": {}}]}
    monkeypatch.setattr(ai_client.requests, "post", lambda *a, **k: FakeResponse(200, body))
    r = ai_client.analyze_video(video, "wave")
    assert r["ok"] and r["job_status"] == "COMPLETED" and "probabilities" not in r["people"][0]


def test_client_bad_video(monkeypatch, video):
    body = {"detail": {"code": "VIDEO_UNREADABLE", "message": "영상 파일을 열 수 없습니다."}}
    monkeypatch.setattr(ai_client.requests, "post", lambda *a, **k: FakeResponse(400, body))
    r = ai_client.analyze_video(video)
    assert not r["ok"] and r["error_message"] == "영상 파일을 열 수 없습니다."


def test_client_server_down_and_missing_file(monkeypatch, video):
    def boom(*a, **k):
        raise requests.ConnectionError()
    monkeypatch.setattr(ai_client.requests, "post", boom)
    assert "연결할 수 없습니다" in ai_client.analyze_video(video)["error_message"]
    assert "찾을 수 없습니다" in ai_client.analyze_video("/no/such.mp4")["error_message"]
