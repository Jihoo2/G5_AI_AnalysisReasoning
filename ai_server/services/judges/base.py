"""
모든 운동 판정기가 공통으로 쓰는 뼈대.

판정 흐름 (운동 종류와 상관없이 같음)
  1) 기준 관절 각도를 프레임마다 계산  (예: 스쿼트 = 무릎 각도)
  2) 각도가 내려갔다 올라오면 '1회'로 셈 → 반복(rep) 나누기
  3) 반복마다 규칙을 검사해 OK / NG + 틀린 이유
  4) OK 비율로 영상 전체 판정

왜 '반복마다' 판정하나요?
  판정 단위는 아직 미정이지만, 반복마다 판정해 두면
  '영상 전체 판정'도, '몇 회 중 몇 회 OK'도 모두 만들 수 있습니다.
  응답에 둘 다 넣어 두고, 화면 담당이 필요한 것만 쓰면 됩니다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import settings
from services.angle_utils import fill_and_smooth


@dataclass
class Rep:
    start: int   # 내려가기 시작한 위치 (배열 인덱스)
    bottom: int  # 가장 낮은 위치
    end: int     # 다시 올라온 위치


def detect_reps(signal: np.ndarray, down: float, up: float, min_len: int = 5) -> list[Rep]:
    """
    각도가 down 아래로 내려갔다가 up 위로 올라오면 1회.

    왜 기준을 두 개(down, up) 쓰나요?
      기준이 하나면 그 값 근처에서 각도가 조금만 흔들려도 내려감/올라옴이 반복돼
      횟수가 부풀려집니다. 두 기준 사이를 '완충 구간'으로 두면 흔들림을 무시할 수 있어요.
    """
    reps: list[Rep] = []
    state = "up"
    start = bottom = 0
    for i, v in enumerate(signal):
        if np.isnan(v):
            continue
        if state == "up" and v < down:
            state, start, bottom = "down", i, i
        elif state == "down":
            if v < signal[bottom]:
                bottom = i
            if v > up:
                if i - start >= min_len:
                    reps.append(Rep(start, bottom, i))
                state = "up"
    return reps


class ExerciseJudge(ABC):
    """운동 하나의 판정기. 새 운동을 추가하려면 이 클래스를 상속해 3가지만 채우면 됩니다."""

    key: str = ""            # API에서 쓰는 이름 (예: "squat")
    display_name: str = ""   # 화면 표시 이름 (예: "스쿼트")

    @abstractmethod
    def main_signal(self, xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """반복을 셀 때 쓰는 기준 각도 (프레임마다)."""

    @abstractmethod
    def rep_thresholds(self) -> tuple[float, float]:
        """(down, up) 기준 각도."""

    @abstractmethod
    def check_rep(self, rep: Rep, xy: np.ndarray, conf: np.ndarray, signal: np.ndarray) -> tuple[dict, list[str]]:
        """반복 1회 검사 → (측정값, 틀린 이유 목록). 이유가 없으면 OK."""

    def rules(self) -> dict:
        """현재 판정 기준 (GET /exercises 에서 팀원이 확인할 수 있도록)."""
        return {}

    def judge(self, xy: np.ndarray, conf: np.ndarray, frame_numbers: list[int]) -> dict:
        signal = fill_and_smooth(self.main_signal(xy, conf))
        down, up = self.rep_thresholds()
        reps = detect_reps(signal, down, up)

        rep_results = []
        for n, rep in enumerate(reps, start=1):
            measures, reasons = self.check_rep(rep, xy, conf, signal)
            rep_results.append({
                "index": n,
                "verdict": "NG" if reasons else "OK",
                "reasons": reasons,
                "start_frame": frame_numbers[rep.start],
                "end_frame": frame_numbers[rep.end],
                "measures": {k: round(float(v), 1) for k, v in measures.items()},
            })

        ok = sum(r["verdict"] == "OK" for r in rep_results)
        total = len(rep_results)
        ok_ratio = ok / total if total else 0.0
        # 영상 전체의 틀린 이유: 반복들에서 나온 이유를 많이 나온 순서로 정리
        reason_counts: dict[str, int] = {}
        for r in rep_results:
            for reason in r["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            "verdict": ("OK" if ok_ratio >= settings.OVERALL_OK_RATIO else "NG") if total else "UNKNOWN",
            "rep_count": total,
            "ok_count": ok,
            "ng_count": total - ok,
            "ok_ratio": round(ok_ratio, 3),
            "reasons": sorted(reason_counts, key=reason_counts.get, reverse=True),
            "reps": rep_results,
        }
