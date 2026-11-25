"""
판정 로직 테스트 — 카메라·YOLO 없이 '가짜 관절 좌표'로 스쿼트·푸시업 판정을 확인합니다.

실행 (ai_server 폴더에서):  python -m pytest tests -q

왜 가짜 데이터로 테스트하나요?
  실제 영상은 촬영 조건마다 결과가 달라서 '로직이 맞는지'를 확인하기 어렵습니다.
  각도를 정확히 아는 가짜 관절을 만들면, 판정 규칙이 의도대로 동작하는지 바로 알 수 있어요.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.angle_utils import angle  # noqa: E402
from services.judges import get_judge  # noqa: E402


def _rotate(v, deg):
    r = np.radians(deg)
    return np.array([v[0] * np.cos(r) - v[1] * np.sin(r), v[0] * np.sin(r) + v[1] * np.cos(r)])


def _point(vertex, toward, deg, length):
    """vertex에서 toward 방향과 deg도 벌어진 곳에 있는 점."""
    u = (toward - vertex) / np.linalg.norm(toward - vertex)
    return vertex + _rotate(u, deg) * length


def squat_frame(knee_deg, hip_deg):
    """옆모습 스쿼트 1프레임: 무릎 각도 knee_deg, 엉덩이 각도 hip_deg."""
    kp = np.zeros((17, 2))
    ankle, knee = np.array([300.0, 600.0]), np.array([300.0, 450.0])
    hip = _point(knee, ankle, knee_deg, 150)
    shoulder = _point(hip, knee, -hip_deg, 180)
    for side in ((5, 11, 13, 15), (6, 12, 14, 16)):
        kp[side[0]], kp[side[1]], kp[side[2]], kp[side[3]] = shoulder, hip, knee, ankle
    kp[7] = kp[8] = shoulder + [30, 40]
    kp[9] = kp[10] = shoulder + [60, 40]
    kp[0] = shoulder + [0, -40]
    return kp


def pushup_frame(elbow_deg, body_deg):
    """옆모습 푸시업 1프레임: 팔꿈치 각도 elbow_deg, 몸(어깨-엉덩이-발목) 각도 body_deg."""
    kp = np.zeros((17, 2))
    wrist, elbow = np.array([200.0, 500.0]), np.array([200.0, 420.0])
    shoulder = _point(elbow, wrist, -elbow_deg, 80)
    ankle = np.array([600.0, 500.0])
    hip = (shoulder + ankle) / 2 + [0, 0]
    # 엉덩이를 아래로 내려 몸 각도를 body_deg로 맞춤
    for dy in np.linspace(0, 150, 1501):
        h = (shoulder + ankle) / 2 + [0, dy]
        if angle(shoulder, h, ankle) <= body_deg:
            hip = h
            break
    for side in ((5, 7, 9, 11, 13, 15), (6, 8, 10, 12, 14, 16)):
        s, e, w, hp, kn, an = side
        kp[s], kp[e], kp[w], kp[hp], kp[an] = shoulder, elbow, wrist, hip, ankle
        kp[kn] = (hip + ankle) / 2
    kp[0] = shoulder + [-30, 0]
    return kp


def _ramp(seq):
    """[(각도, 프레임 수), ...] → 선형으로 이어진 각도 목록."""
    out = []
    for (a, _), (b, n) in zip(seq, seq[1:]):
        out.extend(np.linspace(a, b, n, endpoint=False))
    out.append(seq[-1][0])
    return out


def _run(judge_key, frames):
    xy = np.stack(frames)
    conf = np.full((len(frames), 17), 0.9)
    return get_judge(judge_key).judge(xy, conf, list(range(len(frames))))


def test_angle_basic():
    assert round(angle(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1]))) == 90


def test_squat_two_good_one_shallow():
    # 좋은 스쿼트 2회(무릎 80°) + 얕은 스쿼트 1회(무릎 115°)
    knees = _ramp([(175, 0), (80, 20), (175, 20), (80, 20), (175, 20), (115, 20), (175, 20)])
    res = _run("squat", [squat_frame(k, 75) for k in knees])
    assert res["rep_count"] == 3
    assert [r["verdict"] for r in res["reps"]] == ["OK", "OK", "NG"]
    assert "충분히 앉지 않았어요" in res["reps"][2]["reasons"][0]
    assert res["verdict"] == "NG"  # 2/3 = 0.67 < 0.7


def test_squat_leaning_torso_is_ng():
    knees = _ramp([(175, 0), (80, 20), (175, 20)])
    res = _run("squat", [squat_frame(k, 35) for k in knees])
    assert res["rep_count"] == 1
    assert res["reps"][0]["verdict"] == "NG"
    assert any("상체" in r for r in res["reps"][0]["reasons"])


def test_squat_jitter_not_counted_twice():
    # 120° 근처에서 흔들려도 1회로만 세야 함
    rng = np.random.default_rng(0)
    knees = np.array(_ramp([(175, 0), (80, 25), (175, 25)])) + rng.normal(0, 2, 51)
    res = _run("squat", [squat_frame(k, 75) for k in knees])
    assert res["rep_count"] == 1


def test_standing_still_has_no_rep():
    res = _run("squat", [squat_frame(175, 170) for _ in range(40)])
    assert res["rep_count"] == 0
    assert res["verdict"] == "UNKNOWN"


def test_pushup_good_and_sagging():
    elbows = _ramp([(170, 0), (80, 20), (170, 20), (80, 20), (170, 20)])
    good = [pushup_frame(e, 175) for e in elbows[:41]]
    sag = [pushup_frame(e, 145) for e in elbows[41:]]
    res = _run("pushup", good + sag)
    assert res["rep_count"] == 2
    assert res["reps"][0]["verdict"] == "OK"
    assert res["reps"][1]["verdict"] == "NG"
    assert any("일직선" in r for r in res["reps"][1]["reasons"])
