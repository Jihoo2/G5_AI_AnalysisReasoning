"""
스쿼트 판정 (예시 규칙 — 팀이 기준을 정하면 settings.py의 숫자만 바꾸세요)

  기준 각도 : 무릎 각도 = 엉덩이-무릎-발목
  규칙 1   : 가장 낮을 때 무릎 각도 ≤ SQUAT_DEPTH_MAX  (충분히 앉았나)
  규칙 2   : 가장 낮을 때 엉덩이 각도(어깨-엉덩이-무릎) ≥ SQUAT_TORSO_MIN  (상체가 너무 숙여지지 않았나)

권장 촬영: 옆모습, 전신이 보이게.
"""
import numpy as np

import settings
from services.angle_utils import angle_series, fill_and_smooth

from .base import ExerciseJudge, Rep


class SquatJudge(ExerciseJudge):
    key = "squat"
    display_name = "스쿼트"

    def main_signal(self, xy, conf):
        return angle_series(xy, conf, ("hip", "knee", "ankle"), settings.KEYPOINT_CONF)

    def rep_thresholds(self):
        return settings.SQUAT_DOWN_ANGLE, settings.SQUAT_UP_ANGLE

    def check_rep(self, rep: Rep, xy, conf, signal):
        knee_min = float(np.nanmin(signal[rep.start:rep.end + 1]))
        torso = fill_and_smooth(angle_series(xy, conf, ("shoulder", "hip", "knee"), settings.KEYPOINT_CONF))
        torso_at_bottom = float(torso[rep.bottom])

        reasons = []
        if knee_min > settings.SQUAT_DEPTH_MAX:
            reasons.append(f"충분히 앉지 않았어요 (무릎 최소 {knee_min:.0f}°, 기준 {settings.SQUAT_DEPTH_MAX:.0f}° 이하)")
        if not np.isnan(torso_at_bottom) and torso_at_bottom < settings.SQUAT_TORSO_MIN:
            reasons.append(f"상체가 너무 숙여졌어요 (엉덩이 {torso_at_bottom:.0f}°, 기준 {settings.SQUAT_TORSO_MIN:.0f}° 이상)")
        return {"knee_min_angle": knee_min, "hip_angle_at_bottom": torso_at_bottom}, reasons

    def rules(self):
        return {
            "기준 관절": "무릎 (엉덩이-무릎-발목)",
            "1회 인식": f"무릎 {settings.SQUAT_DOWN_ANGLE:.0f}° 아래로 내려갔다가 {settings.SQUAT_UP_ANGLE:.0f}° 위로 올라옴",
            "깊이": f"가장 낮을 때 무릎 ≤ {settings.SQUAT_DEPTH_MAX:.0f}°",
            "상체": f"가장 낮을 때 엉덩이(어깨-엉덩이-무릎) ≥ {settings.SQUAT_TORSO_MIN:.0f}°",
            "권장 촬영": "옆모습, 전신",
        }
