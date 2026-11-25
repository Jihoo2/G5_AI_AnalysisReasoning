"""
영상 → YOLO Pose + 추적 → 사람(Track ID)별 관절 좌표 모음.

왜 이렇게 나누나요?
  '관절을 뽑는 일'과 '자세가 맞는지 판정하는 일'을 분리해 두면,
  판정 기준이 바뀌어도(미정 사항) 이 파일은 건드리지 않아도 됩니다.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import settings

_pose_model = None
_lock = threading.Lock()  # YOLO 모델은 한 번에 한 요청만 사용 (동시에 쓰면 추적 번호가 섞임)


class VideoReadError(Exception):
    """영상 파일을 열 수 없거나 손상된 경우."""


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class Track:
    """한 사람의 관절 기록."""
    track_id: int
    frames: list[int] = field(default_factory=list)     # 이 사람이 보인 프레임 번호
    xy: list[np.ndarray] = field(default_factory=list)  # 프레임마다 (17, 2) 좌표
    conf: list[np.ndarray] = field(default_factory=list)  # 프레임마다 (17,) 신뢰도

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack(self.xy), np.stack(self.conf)


@dataclass
class PoseResult:
    video: VideoInfo
    frames_processed: int
    tracks: dict[int, Track]


def get_pose_model():
    """YOLO Pose 모델은 무거워서, 처음 한 번만 불러오고 계속 재사용합니다."""
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO(settings.POSE_WEIGHTS)
    return _pose_model


def inspect_video(path: str | Path) -> VideoInfo:
    """영상이 실제로 읽히는지 먼저 확인합니다 (손상 파일과 '사람 없음'을 구분하기 위해)."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("영상 파일을 열 수 없습니다. MP4 등 지원 형식인지 확인하세요.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoReadError("영상에서 프레임을 읽을 수 없습니다. 파일이 손상되었을 수 있습니다.")
        return VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=frame.shape[1],
            height=frame.shape[0],
        )
    finally:
        cap.release()


def extract_tracks(path: str | Path) -> PoseResult:
    video = inspect_video(path)
    model = get_pose_model()
    tracks: dict[int, Track] = {}
    processed = 0

    with _lock:
        # persist=False: 영상마다 새 추적기 → Track ID가 매번 1부터 시작
        results = model.track(source=str(path), stream=True, persist=False, verbose=False,
                              conf=settings.POSE_CONF, tracker=settings.POSE_TRACKER)
        for frame_idx, r in enumerate(results):
            processed += 1
            if processed > settings.MAX_FRAMES:
                break
            if r.boxes is None or r.boxes.id is None or r.keypoints is None:
                continue  # 이 프레임엔 사람이 없거나 아직 추적 번호가 없음
            ids = r.boxes.id.int().cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()
            conf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                    else np.ones(xy.shape[:2], dtype=np.float32))
            for i, tid in enumerate(ids):
                t = tracks.setdefault(int(tid), Track(track_id=int(tid)))
                t.frames.append(frame_idx)
                t.xy.append(xy[i])
                t.conf.append(conf[i])

    return PoseResult(video=video, frames_processed=min(processed, settings.MAX_FRAMES), tracks=tracks)
