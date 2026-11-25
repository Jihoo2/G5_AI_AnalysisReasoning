"""
POST /predict/video — 영상 1건을 받아 사람별 자세 판정(OK / NG)을 JSON으로 돌려줍니다.

요청 : multipart/form-data
  file      (필수) 영상 파일
  exercise  (선택) 운동 종류 — 미정이라 선택값. 없으면 settings.DEFAULT_EXERCISE

응답 status
  OK         1명 이상 판정 완료
  NO_PERSON  영상은 정상인데 사람이 없음
  NO_REP     사람은 있지만 반복 동작을 찾지 못함 (너무 짧거나 동작이 작음)
HTTP 오류 : 400 영상 손상, 413 용량 초과, 415 형식 오류, 422 모르는 운동
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import settings
from services.judges import JUDGES, get_judge
from services.pose_pipeline import VideoReadError, extract_tracks

router = APIRouter(tags=["predictions"])

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _error(status: int, code: str, message: str) -> HTTPException:
    """오류 모양을 하나로 통일: {"detail": {"code": ..., "message": ...}}
    code는 프로그램이 비교하는 값, message는 사람이 읽는 문장입니다."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@router.get("/exercises")
def list_exercises():
    """지원하는 운동과 현재 판정 기준. 팀원이 기준 숫자를 확인할 때 씁니다."""
    return {
        "default": settings.DEFAULT_EXERCISE,
        "overall_ok_ratio": settings.OVERALL_OK_RATIO,
        "exercises": [{"key": j.key, "name": j.display_name, "rules": j.rules()} for j in JUDGES.values()],
    }


@router.post("/predict/video")
def predict_video(
    file: UploadFile = File(..., description="판정할 운동 영상 (MP4 권장, 옆모습·전신)"),
    exercise: str | None = Form(None, description="운동 종류: squat, pushup (비우면 기본값)"),
):
    key = (exercise or settings.DEFAULT_EXERCISE).strip().lower()
    judge = get_judge(key)
    if judge is None:
        raise _error(422, "UNKNOWN_EXERCISE", f"지원하지 않는 운동입니다: {key} (가능: {', '.join(JUDGES)})")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise _error(415, "UNSUPPORTED_FORMAT", f"지원하지 않는 형식입니다: {ext or '(확장자 없음)'}")

    started = time.perf_counter()
    # 업로드 파일을 임시 파일로 저장 → OpenCV·YOLO는 파일 경로로 영상을 읽기 때문
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_MB:
            raise _error(413, "FILE_TOO_LARGE", f"{size_mb:.1f}MB — 최대 {settings.MAX_UPLOAD_MB}MB까지 가능합니다.")
        try:
            pose = extract_tracks(tmp_path)
        except VideoReadError as exc:
            raise _error(400, "VIDEO_UNREADABLE", str(exc)) from exc
    finally:
        file.file.close()
        Path(tmp_path).unlink(missing_ok=True)  # 분석이 끝나면 임시 파일 삭제

    results = []
    for tid in sorted(pose.tracks):
        track = pose.tracks[tid]
        if len(track.frames) < settings.MIN_TRACK_FRAMES:
            continue  # 화면에 잠깐 스친 사람은 제외
        xy, conf = track.arrays()
        results.append({"track_id": tid, "frame_count": len(track.frames), **judge.judge(xy, conf, track.frames)})

    judged = [r for r in results if r["rep_count"] > 0]
    if judged:
        status, message = "OK", f"{len(judged)}명의 {judge.display_name} 자세를 판정했습니다."
    elif pose.tracks:
        status, message = "NO_REP", f"사람은 찾았지만 {judge.display_name} 반복 동작을 찾지 못했습니다. 옆모습·전신이 보이게, 2회 이상 촬영해 주세요."
    else:
        status, message = "NO_PERSON", "영상에서 사람을 찾지 못했습니다."

    return {
        "status": status,
        "message": message,
        "exercise": {"key": judge.key, "name": judge.display_name},
        "results": results,
        "frames_processed": pose.frames_processed,
        "video": pose.video.__dict__,
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
        "judge": {"type": "rule", "version": "0.1"},
    }
