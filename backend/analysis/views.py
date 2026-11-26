from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response


def index(request):
    """/analyses/ 로 들어오면 분석 화면(screen.html)을 돌려준다."""
    return render(request, "analysis/screen.html")


@api_view(["POST"])                       # DRF: POST 요청만 받음
def analyze(request):
    """STEP 4 — JS 가 보낸 영상이 Django 에 도착했는지 확인만 한다."""
    video = request.FILES.get("video")    # JS 의 FormData "video"
    if video is None:
        return Response({"error": "video 파일이 없습니다."}, status=400)

    print("받은 영상:", video.name, video.size, "bytes")   # 터미널에서 확인
    return Response({"file_name": video.name, "size": video.size})