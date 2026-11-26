from django.urls import path

from . import views

urlpatterns = [
    path("analyze/", views.analyze, name="analyze"),            # POST /api/v1/analyze/
    path("analyze/<int:pk>/", views.result, name="result"),     # GET  /api/v1/analyze/<id>/
]