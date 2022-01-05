import cv2
import threading
import numpy as np
import socket

# Django
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

# Model
from models.face_detection import SY_COUNT, face_detect, sy_detection
from models.sleep_detection import sleep_detect

# Use server address. localhost
HOST = '172.30.1.26'

# PORT Number
PORT = 9999

# Generate Socket Object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Handle WinError 10048
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind for Connecting Socket
server_socket.bind((HOST, PORT))

# Permission for Access
server_socket.listen()

# Return Socket
client_socket, addr = server_socket.accept()

# Client Address
print('Connected by', addr)


# Frame Generator
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Identification Page
class FaceCamera(object):
    def __init__(self):
        # Cam Load
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def get_frame(self):
        global SY_COUNT
        image = self.frame
        cam = self.video

        # Execute Face Detection
        face_detect(cam, image) # model function
        from models.face_detection import SY_COUNT

        # Call GiGA-Genie        
        if SY_COUNT == 100:
            client_socket.sendall("안녕하세요 승용님, 학습을 시작하겠습니다.".encode())
            SY_COUNT = 101

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


# Course Page Cam Load
TEMP_CAP = cv2.VideoCapture(0) # Sleep Detect Cam
TEMP_CAP2 = cv2.VideoCapture(0) # Sleep Detect Cam (To avoid conflict) 

# Lecture Page
class VideoCamera(object):
    def __init__(self):
        # Cam Load
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def get_frame(self):
        global image
        global TEMP_CAP
        global TEMP_CAP2
        global YAWN_COUNTER
        global client_socket
        from models.sleep_detection import YAWN_COUNTER

        image = self.frame
        if type(image) != np.ndarray:
            _, image = TEMP_CAP.read()
        try:
            image = cv2.flip(image, 1)
            sleep_detect(image)
        except:
            TEMP_CAP.release()
            _, image = TEMP_CAP2.read()
            image = cv2.flip(image, 1)
            sleep_detect(image)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

# Identification Page Render
def index(request):
    return render(request, "core/index.html")

# Zip Streaming Video
@gzip.gzip_page
def cam_test(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("Error")
        pass

# Zip Streaming Video
@gzip.gzip_page
def face_detection(request):
    try:
        cam = FaceCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("Error")
        pass