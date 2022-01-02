
import cv2
import threading
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Django
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

# Model
from models.face_detection import face_detect
from models.sleep_detection import sleep_detect
# from models.pose_detection import pose_detect
 

# Frame Generator
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Index Page Cam Load
class FaceCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def get_frame(self):
        image = self.frame
        cam = self.video
        face_detect(cam, image) # model function
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


# Course Page Cam Load
class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def get_frame(self):
        global image
        image = self.frame
        sleep_detect(image)
        # jpeg encoding
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

# Pose Page Cam Load

class PoseCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
    
    def get_frame(self):
        image = self.frame
        cam = self.video
        # pose_detect(cam, image) # model function
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


# MAIN PAGE RENDER
def index(request):
    return render(request, "core/index.html")


@gzip.gzip_page
def cam_test(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("Error")
        pass


@gzip.gzip_page
def face_detection(request):
    try:
        cam = FaceCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("Error")
        pass


@gzip.gzip_page
def pose_detection(request):
    try:
        cam = PoseCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("Error")
        pass