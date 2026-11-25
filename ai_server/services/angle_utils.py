"""
관절 좌표 → 각도 계산 도구.

COCO 17 Keypoint 번호 (YOLO Pose가 돌려주는 순서)
 0 코, 1 왼눈, 2 오른눈, 3 왼귀, 4 오른귀,
 5 왼어깨, 6 오른어깨, 7 왼팔꿈치, 8 오른팔꿈치, 9 왼손목, 10 오른손목,
 11 왼엉덩이, 12 오른엉덩이, 13 왼무릎, 14 오른무릎, 15 왼발목, 16 오른발목

왜 각도를 쓰나요?
  좌표(픽셀 위치)는 카메라와의 거리나 사람 키에 따라 달라지지만,
  관절 각도는 사람이 가까이 있든 멀리 있든 같은 자세면 같은 값이 나옵니다.
  그래서 "무릎을 충분히 굽혔나?" 같은 자세 판정에 잘 맞습니다.
"""
from __future__ import annotations

import numpy as np

# 왼쪽/오른쪽 관절 번호
SIDES = {
    "left":  {"shoulder": 5, "elbow": 7, "wrist": 9,  "hip": 11, "knee": 13, "ankle": 15},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "knee": 14, "ankle": 16},
}


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b를 꼭짓점으로 하는 각도 (0~180도). 예: angle(엉덩이, 무릎, 발목) = 무릎 각도."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def angle_series(xy: np.ndarray, conf: np.ndarray, joints: tuple[str, str, str],
                 min_conf: float) -> np.ndarray:
    """
    프레임마다 지정한 세 관절의 각도를 계산합니다.
    왼쪽·오른쪽 중 더 잘 보이는(신뢰도가 높은) 쪽을 프레임마다 골라 씁니다.
    → 옆에서 찍은 영상은 한쪽 다리가 가려지기 때문입니다.

    xy   : (T, 17, 2)   conf : (T, 17)
    반환 : (T,) 각도. 관절이 안 보이면 NaN
    """
    out = np.full(len(xy), np.nan, dtype=np.float32)
    for t in range(len(xy)):
        best, best_conf = None, -1.0
        for side in SIDES.values():
            idx = [side[j] for j in joints]
            c = float(conf[t, idx].min())
            if c > best_conf:
                best, best_conf = idx, c
        if best_conf >= min_conf:
            a, b, c = (xy[t, i] for i in best)
            out[t] = angle(a, b, c)
    return out


def fill_and_smooth(series: np.ndarray, window: int = 5) -> np.ndarray:
    """
    잠깐 안 보인 프레임(NaN)은 앞뒤 값으로 채우고, 이동평균으로 흔들림을 줄입니다.
    왜? Pose 추출은 프레임마다 조금씩 떨려서, 그대로 쓰면 1번 앉았는데 2번으로 셀 수 있습니다.
    """
    s = np.asarray(series, dtype=np.float32).copy()
    valid = ~np.isnan(s)
    if valid.sum() < 2:
        return s
    idx = np.arange(len(s))
    s = np.interp(idx, idx[valid], s[valid]).astype(np.float32)
    if window > 1 and len(s) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        pad = window // 2
        s = np.convolve(np.pad(s, pad, mode="edge"), kernel, mode="valid").astype(np.float32)
    return s
