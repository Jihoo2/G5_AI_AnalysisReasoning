"""
POST /predict/video — 영상 1건 → 사람별 동작 인식 + OK/NG

요청 (multipart/form-data)
  file             (필수) 영상
  expected_action  (선택) 사용자가 하려던 동작. 주면 OK/NG까지 판정

응답 status:  OK / NO_PERSON(사람 없음) / TOO_SHORT(사람은 있지만 판단할 만큼 길지 않음)
HTTP 오류:    400 영상 손상 · 413 용량 초과 · 415 형식 · 422 모르는 동작 · 503 모델 준비 안 됨
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import app_config as cfg
from services import model_loader
from services.inference import judge, predict_track
from services.pose_pipeline import VideoReadError, extract

router = APIRouter(tags=["predictions"])
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _error(status: int, code: str, message: str) -> HTTPException:
    """오류 모양 통일: code는 프로그램이 비교하는 값, message는 사람이 읽는 문장."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@router.get("/actions")
def actions():
    """모델이 인식할 수 있는 동작 목록과 판정 기준 (팀원 확인용)."""
    lm = model_loader.get()
    return {
        "actions": lm.classes if lm else [],
        "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
        "window_seconds": cfg.WINDOW_SECONDS,
    }


@router.post("/predict/video")
def predict_video(
    file: UploadFile = File(..., description="분석할 영상 (MP4 권장)"),
    expected_action: str | None = Form(None, description="하려던 동작 (예: wave). 비우면 인식만 하고 OK/NG는 생략"),
):
    lm = model_loader.get()
    if lm is None:
        raise _error(503, "MODEL_NOT_READY", model_loader.error() or "모델이 준비되지 않았습니다.")

    expected = (expected_action or "").strip() or None
    if expected == "string":  # /docs 화면이 자동으로 채우는 예시값은 '없음'으로 취급
        expected = None
    if expected and expected not in lm.classes:
        raise _error(422, "UNKNOWN_ACTION", f"모르는 동작입니다: {expected} (가능: {', '.join(lm.classes)})")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise _error(415, "UNSUPPORTED_FORMAT", f"지원하지 않는 형식입니다: {ext or '(확장자 없음)'}")

    started = time.perf_counter()
    fd, tmp = tempfile.mkstemp(suffix=ext)  # OpenCV·YOLO는 파일 경로로 읽으므로 임시 저장
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)
        size_mb = os.path.getsize(tmp) / (1024 * 1024)
        if size_mb > cfg.MAX_UPLOAD_MB:
            raise _error(413, "FILE_TOO_LARGE", f"{size_mb:.1f}MB — 최대 {cfg.MAX_UPLOAD_MB}MB")
        try:
            pose = extract(tmp)
        except VideoReadError as exc:
            raise _error(400, "VIDEO_UNREADABLE", str(exc)) from exc
    finally:
        file.file.close()
        Path(tmp).unlink(missing_ok=True)

    results = []
    for tid in sorted(pose.tracks):
        pred = predict_track(pose.tracks[tid], pose.fps, lm)
        if pred:
            results.append({"track_id": tid, **pred, **judge(pred, expected)})

    if results:
        status, message = "OK", f"{len(results)}명의 동작을 분석했습니다."
    elif pose.tracks:
        status, message = "TOO_SHORT", "사람은 찾았지만 판단할 만큼 오래·선명하게 보이지 않았습니다."
    else:
        status, message = "NO_PERSON", "영상에서 사람을 찾지 못했습니다."

    return {
        "status": status,
        "message": message,
        "expected_action": expected,
        "results": results,
        "video": {"fps": pose.fps, "width": pose.width, "height": pose.height, "frame_count": pose.frame_count},
        "frames_processed": pose.frames_processed,
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
    }
