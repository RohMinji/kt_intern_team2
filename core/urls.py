from django.urls import path
from . import views
from models import face_detection, sleep_detection

app_name = "core"

urlpatterns = [
    path("", views.index, name="main"),
    path("sy-detection/", face_detection.sy_detection, name="sy_detection"),
    path("video-stop/",  sleep_detection.videoStop, name="videoStop"),
    path("cam/", views.cam_test, name="cam_test"),
    path("face-detection/", views.face_detection, name="face_detection"),
    path("pose-detection/", views.pose_detection, name="pose_detection"),
]