from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import InferenceRecord


def index(request):
    """/analyses/ 로 들어오면 분석 화면(screen.html)을 돌려준다."""
    return render(request, "analysis/screen.html")


@api_view(["POST"])                       # DRF: POST 요청만 받음
def analyze(request):
    """STEP 5 — 받은 영상을 저장하고 기록 1건(RECEIVED)을 만든다."""
    video = request.FILES.get("video")    # JS 의 FormData "video"
    if video is None:
        return Response({"error": "video 파일이 없습니다."}, status=400)

    record = InferenceRecord.objects.create(video_file=video)   # 파일 → media/videos/, 경로 → DB
    return Response({"id": record.id, "status": record.status})