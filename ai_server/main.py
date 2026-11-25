"""
FastAPI 자세 판정 서버 진입점.
  실행: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
  문서: http://localhost:8001/docs

이 서버는 '판정'만 합니다. DB 저장·회원 관리는 Django(backend/)가 맡습니다.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

import settings
from routers import predictions
from services.judges import JUDGES
from services.pose_pipeline import get_pose_model

_state = {"pose_ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 YOLO Pose를 미리 한 번 불러둡니다.
    # 왜? 첫 요청 때 불러오면 그 요청만 유난히 느려서 Celery가 타임아웃으로 오해할 수 있어요.
    try:
        get_pose_model()
        _state["pose_ready"] = True
    except Exception as exc:  # noqa: BLE001 — 실패해도 서버는 켜고 /health에서 원인을 보여줌
        _state["error"] = f"{exc.__class__.__name__}: {exc}"
    yield


app = FastAPI(title="G5 자세 판정 AI 서버", version="0.1.0", lifespan=lifespan)
app.include_router(predictions.router)


@app.get("/health", tags=["health"])
def health():
    """서버가 살아 있는지, Pose 모델이 준비됐는지 확인 (Docker healthcheck에도 사용)."""
    return {
        "status": "ok" if _state["pose_ready"] else "error",
        "pose_model_ready": _state["pose_ready"],
        "pose_weights": settings.POSE_WEIGHTS,
        "exercises": list(JUDGES),
        "default_exercise": settings.DEFAULT_EXERCISE,
        "judge_type": "rule",
        "error": _state["error"],
    }
