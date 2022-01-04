import cv2
import threading
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import pandas as pd

# Django
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

# Model
from models.face_detection import face_detect
from models.sleep_detection import sleep_detect
 

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
TEMP_CAP = cv2.VideoCapture(0)
TEMP_CAP2 = cv2.VideoCapture(0)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.create_Data()
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def get_frame(self):
        global image
        global TEMP_CAP
        global TEMP_CAP2
        image = self.frame
        if type(image) != np.ndarray:
            _, image = TEMP_CAP.read()
        try:
            sleep_detect(image)
        except:
            TEMP_CAP.release()
            _, image = TEMP_CAP2.read()
            sleep_detect(image)
        # jpeg encoding
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
    
    def create_Data(self):

        txtDF= pd.DataFrame(columns=['label','time'])
        imgDF= pd.DataFrame(columns=['ear','time'])

        txtDF.to_csv('static/data/txtDF.csv',header = True, index=False)
        imgDF.to_csv('static/data/imgDF.csv',header = True, index=False)

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