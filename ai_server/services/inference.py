"""
사람별 프레임 Feature → Window(5초)로 자르기 → 30 Step → Random Forest → 동작 + 신뢰도 → OK/NG.

7일차 실시간 설정(day07_realtime.json)을 영상 파일에 맞게 적용합니다.
  window_seconds 5.0          → 5초 분량을 한 번에 판단
  prediction_interval_sec 0.25 → 0.25초씩 밀면서 여러 번 판단
  min_window_fill_ratio 0.8    → 5초 중 80% 이상 사람이 보여야 그 Window를 사용
  min_valid_feature_ratio 0.6  → 숫자 중 60% 이상이 유효(가려지지 않음)해야 사용
  confidence_threshold 0.55    → 이보다 낮으면 '판단 불확실'

왜 여러 Window의 평균을 쓰나요?
  한 번만 보면 순간적인 흔들림에 결과가 바뀝니다. 여러 번 본 확률을 평균 내면 안정적입니다.
  (7일차의 '예측 안정화'와 같은 목적)
"""
from __future__ import annotations

import numpy as np

import app_config as cfg
from services.features import flatten, to_sequence
from services.model_loader import LoadedModel
from services.pose_pipeline import Track


def make_windows(frame_idx: list[int], fps: float, min_valid_frames: int) -> list[np.ndarray]:
    """프레임 번호 목록 → Window마다 '이 Track 배열의 몇 번째들을 쓸지' 목록."""
    idx = np.asarray(frame_idx)
    win = max(1, round(cfg.WINDOW_SECONDS * fps))
    stride = max(1, round(cfg.WINDOW_STRIDE_SECONDS * fps))
    first, last = int(idx[0]), int(idx[-1])

    if last - first + 1 < win:  # 영상(또는 등장 시간)이 5초보다 짧으면 전체를 1개 Window로
        return [np.arange(len(idx))] if len(idx) >= min_valid_frames else []

    starts = list(range(first, last - win + 2, stride))
    if starts[-1] + win - 1 < last:  # 끝부분이 빠지지 않도록 마지막 Window를 영상 끝에 맞춤
        starts.append(last - win + 1)
    windows = []
    for start in starts:
        sel = np.where((idx >= start) & (idx < start + win))[0]
        if len(sel) >= min_valid_frames and len(sel) / win >= cfg.MIN_WINDOW_FILL_RATIO:
            windows.append(sel)
    return windows


def predict_track(track: Track, fps: float, lm: LoadedModel) -> dict | None:
    feats = np.stack(track.features)
    X = []
    for sel in make_windows(track.frame_idx, fps, lm.min_valid_frames):
        chunk = feats[sel]
        if np.mean(~np.isnan(chunk)) < cfg.MIN_VALID_FEATURE_RATIO:
            continue  # 관절이 너무 많이 가려진 구간은 건너뜀
        X.append(flatten(to_sequence(chunk, lm.steps), lm.columns))
    if not X:
        return None

    proba = lm.model.predict_proba(np.stack(X))
    mean = proba.mean(axis=0)
    best = int(np.argmax(mean))
    confident = proba.max(axis=1) >= cfg.CONFIDENCE_THRESHOLD
    votes = np.bincount(proba.argmax(axis=1)[confident], minlength=len(lm.classes))
    return {
        "action_label": lm.classes[best],
        "confidence": round(float(mean[best]), 4),
        "is_uncertain": bool(mean[best] < cfg.CONFIDENCE_THRESHOLD),
        "window_count": len(X),
        "frame_count": len(track.frame_idx),
        "votes": {c: int(v) for c, v in zip(lm.classes, votes) if v},
        "probabilities": {c: round(float(p), 4) for c, p in zip(lm.classes, mean)},
    }


def judge(pred: dict, expected_action: str | None) -> dict:
    """
    OK/NG 판정 (임시 규칙 — 팀에서 방식이 정해지면 이 함수만 바꾸면 됨)
      사용자가 하려던 동작(expected_action)을 AI가 같은 동작으로, 신뢰도 기준 이상으로 인식하면 OK.
      expected_action 이 없으면 판정하지 않음(None).
    """
    if not expected_action:
        return {"verdict": None, "reason": "판정할 동작(expected_action)이 지정되지 않았습니다."}
    if pred["action_label"] != expected_action:
        return {"verdict": "NG", "reason": f"{expected_action} 대신 {pred['action_label']}(으)로 인식됐습니다."}
    if pred["is_uncertain"]:
        return {"verdict": "NG", "reason": f"동작은 맞지만 신뢰도 {pred['confidence']:.0%}가 기준 {cfg.CONFIDENCE_THRESHOLD:.0%} 미만입니다."}
    return {"verdict": "OK", "reason": f"{expected_action} 동작을 신뢰도 {pred['confidence']:.0%}로 인식했습니다."}
