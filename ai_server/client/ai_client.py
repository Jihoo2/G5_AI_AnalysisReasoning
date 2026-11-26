"""
백엔드(Celery)용 AI 서버 호출 함수 — 이 파일을 backend/ 의 앱 폴더에 그대로 복사해서 쓰세요.

    from .ai_client import analyze_video

    result = analyze_video("/app/media/uploads/clip.mp4", expected_action="wave")
    if result["ok"]:
        job.status = "COMPLETED"
        job.result_text = result["summary"]["display_text"]     # 화면에 그대로 표시
        job.verdict = result["summary"]["verdict"]              # "OK" / "NG" / None
        for p in result["people"]:
            PersonPrediction.objects.create(job=job, **p)       # 필드 이름이 그대로 맞음
    else:
        job.status = "FAILED"
        job.error_message = result["error_message"]             # 사용자에게 보여줄 문장
    job.save()

왜 이 파일을 FastAPI 담당이 만들어 주나요?
  AI 서버의 응답 모양과 오류 종류는 FastAPI 담당이 가장 잘 압니다.
  연결 실패·시간 초과·영상 손상 같은 경우를 여기서 전부 정리해 두면,
  백엔드는 ok 값 하나로 성공/실패만 나누면 됩니다.

필요 패키지: requests
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

# Docker 안: http://fastapi:8001  /  내 PC에서 테스트: http://localhost:8001
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")
AI_REQUEST_TIMEOUT = int(os.getenv("AI_REQUEST_TIMEOUT", "300"))

# PersonPrediction 에 저장할 필드 (응답에서 이 값들만 골라 줌)
PERSON_FIELDS = ("track_id", "action_label", "action_name", "confidence",
                 "is_uncertain", "verdict", "reason", "window_count", "frame_count")


def _fail(message: str, raw: dict | None = None) -> dict:
    return {"ok": False, "job_status": "FAILED", "error_message": message,
            "summary": None, "people": [], "processing_time_ms": None, "raw": raw}


def analyze_video(video_path: str | Path, expected_action: str | None = None) -> dict:
    """
    영상 파일을 AI 서버에 보내고, DB에 바로 저장할 수 있는 모양으로 돌려줍니다.

    반환값
      ok                 True = 분석 완료(사람이 없어도 완료), False = 실패
      job_status         "COMPLETED" 또는 "FAILED" — AnalysisJob.status 에 그대로
      error_message      실패 이유 (사용자에게 보여줄 문장)
      summary            대표 1명 결과: action_name, confidence_percent, verdict, display_text ...
      people             사람별 결과 목록 — PersonPrediction 한 줄씩
      processing_time_ms AI 처리 시간
      raw                AI 서버 원본 응답 (디버깅용)
    """
    url = f"{AI_SERVICE_URL.rstrip('/')}/predict/video"
    path = Path(video_path)
    data = {"expected_action": expected_action} if expected_action else {}
    try:
        with path.open("rb") as f:
            res = requests.post(url, files={"file": (path.name, f, "video/mp4")},
                                data=data, timeout=AI_REQUEST_TIMEOUT)
    except FileNotFoundError:
        return _fail(f"분석할 영상 파일을 찾을 수 없습니다: {path.name}")
    except requests.Timeout:
        return _fail(f"AI 서버 응답 시간이 초과됐습니다 ({AI_REQUEST_TIMEOUT}초).")
    except requests.ConnectionError:
        return _fail(f"AI 서버에 연결할 수 없습니다 ({AI_SERVICE_URL}). FastAPI가 켜져 있는지 확인하세요.")

    try:
        body = res.json()
    except ValueError:
        return _fail(f"AI 서버 응답을 읽을 수 없습니다 (HTTP {res.status_code}).")

    if res.status_code != 200:  # 400 영상 손상, 413 용량, 415 형식, 422 모르는 동작, 503 모델 없음
        detail = body.get("detail")
        msg = detail.get("message") if isinstance(detail, dict) else str(detail)
        return _fail(msg or f"AI 서버 오류 (HTTP {res.status_code})", body)

    # 200 이면 사람이 없거나(NO_PERSON) 너무 짧아도(TOO_SHORT) '분석 완료' — summary.display_text 에 안내 문장이 들어 있음
    return {
        "ok": True,
        "job_status": "COMPLETED",
        "error_message": "",
        "summary": body.get("summary"),
        "people": [{k: r.get(k) for k in PERSON_FIELDS} for r in body.get("results", [])],
        "processing_time_ms": body.get("processing_time_ms"),
        "raw": body,
    }
