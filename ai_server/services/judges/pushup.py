"""
푸시업 판정 (예시 규칙 — 팀이 기준을 정하면 settings.py의 숫자만 바꾸세요)

  기준 각도 : 팔꿈치 각도 = 어깨-팔꿈치-손목
  규칙 1   : 가장 낮을 때 팔꿈치 각도 ≤ PUSHUP_DEPTH_MAX  (충분히 내려갔나)
  규칙 2   : 반복 내내 몸 각도(어깨-엉덩이-발목) ≥ PUSHUP_BODY_MIN  (몸이 일직선인가)

권장 촬영: 옆모습, 전신.
"""
import numpy as np

import settings
from services.angle_utils import angle_series, fill_and_smooth

from .base import ExerciseJudge, Rep


class PushupJudge(ExerciseJudge):
    key = "pushup"
    display_name = "푸시업"

    def main_signal(self, xy, conf):
        return angle_series(xy, conf, ("shoulder", "elbow", "wrist"), settings.KEYPOINT_CONF)

    def rep_thresholds(self):
        return settings.PUSHUP_DOWN_ANGLE, settings.PUSHUP_UP_ANGLE

    def check_rep(self, rep: Rep, xy, conf, signal):
        elbow_min = float(np.nanmin(signal[rep.start:rep.end + 1]))
        body = fill_and_smooth(angle_series(xy, conf, ("shoulder", "hip", "ankle"), settings.KEYPOINT_CONF))
        body_min = float(np.nanmin(body[rep.start:rep.end + 1])) if not np.isnan(body).all() else float("nan")

        reasons = []
        if elbow_min > settings.PUSHUP_DEPTH_MAX:
            reasons.append(f"충분히 내려가지 않았어요 (팔꿈치 최소 {elbow_min:.0f}°, 기준 {settings.PUSHUP_DEPTH_MAX:.0f}° 이하)")
        if not np.isnan(body_min) and body_min < settings.PUSHUP_BODY_MIN:
            reasons.append(f"몸이 일직선이 아니에요 — 엉덩이가 처지거나 들렸어요 (최소 {body_min:.0f}°, 기준 {settings.PUSHUP_BODY_MIN:.0f}° 이상)")
        return {"elbow_min_angle": elbow_min, "body_min_angle": body_min}, reasons

    def rules(self):
        return {
            "기준 관절": "팔꿈치 (어깨-팔꿈치-손목)",
            "1회 인식": f"팔꿈치 {settings.PUSHUP_DOWN_ANGLE:.0f}° 아래로 내려갔다가 {settings.PUSHUP_UP_ANGLE:.0f}° 위로 올라옴",
            "깊이": f"가장 낮을 때 팔꿈치 ≤ {settings.PUSHUP_DEPTH_MAX:.0f}°",
            "자세": f"반복 내내 몸(어깨-엉덩이-발목) ≥ {settings.PUSHUP_BODY_MIN:.0f}°",
            "권장 촬영": "옆모습, 전신",
        }
