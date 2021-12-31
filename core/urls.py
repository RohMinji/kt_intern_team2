from django.urls import path
from . import views

app_name = "core"

urlpatterns = [
    path("", views.index, name="main"),
    path("cam/", views.cam_test, name="cam_test"),
    path("face_detection/", views.face_detection, name="face_detection"),
]