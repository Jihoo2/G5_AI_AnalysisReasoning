from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("analyses/", include("analysis.urls")),   # /analyses/ 로 시작 → 앱 urls.py 로
]