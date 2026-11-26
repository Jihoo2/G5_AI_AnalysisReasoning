from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),   # /analyses/ → views.index
]