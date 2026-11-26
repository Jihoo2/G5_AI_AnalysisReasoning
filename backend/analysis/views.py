import requests
from django.shortcuts import get_object_or_404, render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import InferenceRecord

FASTAPI_URL = "http://localhost:8001/predict/video"   # 효근님 FastAPI 주소


def index(request):
    """/analyses/ 로 들어오면 분석 화면(screen.html)을 돌려준다."""
    return render(request, "analysis/screen.html")


@api_view(["POST"])                       # DRF: POST 요청만 받음
def analyze(request):
    """영상 받기 → 기록 저장(STEP 5) → FastAPI 판정 요청 → 결과 기록(STEP 8)."""
    video = request.FILES.get("video")    # JS 의 FormData "video"
    if video is None:
        return Response({"error": "video 파일이 없습니다."}, status=400)

    record = InferenceRecord.objects.create(video_file=video)   # 파일 → media/videos/, 경로 → DB

    # ── STEP 8: FastAPI 에 영상을 보내고, 돌아온 OK/NG 를 그대로 기록
    try:
        with record.video_file.open("rb") as f:
            res = requests.post(
                FASTAPI_URL,
                params={"detail": "true"},                 # JSON 으로 받기 (없으면 "OK" 글자만 옴)
                files={"file": (video.name, f)},           # FastAPI 는 "file" 로 받음, 원래 파일명 그대로
                data={"expected_action": request.POST.get("action", "")},  # 화면에서 고른 동작 (없으면 파일명에서 읽음)
                timeout=120,
            )
        res.raise_for_status()                             # 4xx·5xx 면 아래 except 로
        record.verdict = res.json()["verdict"]             # 판정은 FastAPI — Django 는 받아 적기만
        record.status = "DONE"
    except requests.RequestException:                      # FastAPI 꺼짐·시간 초과·422 등
        record.status = "FAILED"
    record.save()

    return Response(_result(record))                       # ★ 응답 모양을 _result() 한 곳에서


@api_view(["GET"])
def result(request, pk):
    """STEP 9 — 기록 id 로 DB 에 저장된 결과를 조회한다. 없는 id 면 404."""
    record = get_object_or_404(InferenceRecord, pk=pk)
    return Response(_result(record))


def _result(record):
    """POST(분석)와 GET(조회)이 같은 모양으로 돌려주는 결과 — 한 곳에서만 정의"""
    return {"id": record.id, "status": record.status, "verdict": record.verdict}