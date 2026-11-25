"""
FastAPI AI 서버 진입점.
  실행: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
  문서: http://localhost:8001/docs

이 서버는 '영상 → 판정'만 합니다. DB 저장·회원·화면은 Django(backend/)가 맡습니다.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

import app_config as cfg
from routers import predictions
from services import model_loader
from services.pose_pipeline import get_pose_model

_pose = {"ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 모델 두 개(Random Forest, YOLO Pose)를 미리 불러둠
    # 왜? 첫 요청 때 불러오면 그 요청만 유난히 느려져 Celery가 타임아웃으로 오해할 수 있음
    model_loader.init()
    try:
        get_pose_model()
        _pose["ready"] = True
    except Exception as exc:  # noqa: BLE001
        _pose["error"] = f"{exc.__class__.__name__}: {exc}"
    yield


app = FastAPI(title="G5 AI 서버", version="0.2.0", lifespan=lifespan)
app.include_router(predictions.router)


@app.get("/health", tags=["health"])
def health():
    """서버·모델 준비 상태 (Docker healthcheck에도 사용)."""
    lm = model_loader.get()
    ready = lm is not None and _pose["ready"]
    return {
        "status": "ok" if ready else "error",
        "rf_model_ready": lm is not None,
        "pose_model_ready": _pose["ready"],
        "pose_weights": cfg.POSE_WEIGHTS.name,
        "classes": lm.classes if lm else [],
        "feature_count": len(lm.columns) if lm else 0,
        "sequence_steps": lm.steps if lm else None,
        "warnings": lm.warnings if lm else [],
        "errors": [e for e in (model_loader.error(), _pose["error"]) if e],
    }
