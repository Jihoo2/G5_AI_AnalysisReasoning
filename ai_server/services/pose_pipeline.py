"""
영상 → YOLO Pose + 추적 → 사람(Track ID)별 프레임 Feature(38개씩).

설정은 project.json(Pose)과 day08_multi_person.json(추적기)에서 가져옵니다.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import app_config as cfg
from services.features import frame_features

_pose = None
_lock = threading.Lock()  # YOLO는 한 번에 한 요청만 (동시에 쓰면 추적 번호가 섞임)


class VideoReadError(Exception):
    """영상을 열 수 없거나 손상된 경우."""


@dataclass
class Track:
    track_id: int
    frame_idx: list[int] = field(default_factory=list)          # 이 사람이 보인 프레임 번호
    features: list[np.ndarray] = field(default_factory=list)    # 프레임마다 38개


@dataclass
class PoseResult:
    fps: float
    width: int
    height: int
    frame_count: int
    frames_processed: int
    tracks: dict[int, Track]


def get_pose_model():
    global _pose
    if _pose is None:
        from ultralytics import YOLO
        weights = cfg.POSE_WEIGHTS if cfg.POSE_WEIGHTS.exists() else cfg.POSE_WEIGHTS.name  # 없으면 자동 다운로드
        _pose = YOLO(str(weights))
    return _pose


def inspect_video(path: str | Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("영상 파일을 열 수 없습니다. MP4 등 지원 형식인지 확인하세요.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoReadError("영상에서 프레임을 읽을 수 없습니다. 파일이 손상되었을 수 있습니다.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or cfg.DEFAULT_FPS
        return fps, frame.shape[1], frame.shape[0], int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def extract(path: str | Path) -> PoseResult:
    fps, w, h, n = inspect_video(path)
    model = get_pose_model()
    tracks: dict[int, Track] = {}
    processed = 0
    with _lock:
        results = model.track(
            source=str(path), stream=True, persist=False, verbose=False,
            conf=cfg.PERSON_CONF, imgsz=cfg.IMAGE_SIZE, tracker=cfg.TRACKER, device=cfg.DEVICE,
        )
        for i, r in enumerate(results):
            processed += 1
            if processed > cfg.MAX_FRAMES:
                break
            if r.boxes is None or r.boxes.id is None or r.keypoints is None:
                continue
            ids = r.boxes.id.int().cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()
            conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
            for k, tid in enumerate(ids):
                t = tracks.setdefault(int(tid), Track(int(tid)))
                t.frame_idx.append(i)
                t.features.append(frame_features(xy[k], conf[k], cfg.KEYPOINT_CONF))
    return PoseResult(fps, w, h, n, min(processed, cfg.MAX_FRAMES), tracks)
