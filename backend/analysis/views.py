from django.shortcuts import render


def index(request):
    """/analyses/ 로 들어오면 분석 화면(screen.html)을 돌려준다."""
    return render(request, "analysis/screen.html")