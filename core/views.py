import cv2
import threading
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import pandas as pd
import socket

# Django
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

# Model
from models.face_detection import SY_COUNT, face_detect, sy_detection
from models.sleep_detection import sleep_detect
# from models.pose_detection import pose_detect


# 접속할 서버 주소. localhost 사용
HOST = '172.30.1.21'

# 클라이언트 접속을 대기하는 포트 번호
PORT = 9999       

# 소켓 객체를 생성합니다. 
# 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다. 
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트 사용중이라 연결할 수 없다는 
# WinError 10048 에러 해결를 위해 필요합니다. 
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


# bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는데 사용됩니다.
# HOST는 hostname, ip address, 빈 문자열 ""이 될 수 있습니다.
# 빈 문자열이면 모든 네트워크 인터페이스로부터의 접속을 허용합니다. 
# PORT는 1-65535 사이의 숫자를 사용할 수 있습니다.  
server_socket.bind((HOST, PORT))

# 서버가 클라이언트의 접속을 허용하도록 합니다. 
server_socket.listen()

# accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴합니다. 
client_socket, addr = server_socket.accept()

# 접속한 클라이언트의 주소입니다.
print('Connected by', addr)

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
        global SY_COUNT
        image = self.frame
        cam = self.video
        face_detect(cam, image) # model function
        from models.face_detection import SY_COUNT
        if SY_COUNT == 100:
            client_socket.sendall("안녕하세요 승용님, 학습을 시작하겠습니다.".encode())
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
        global YAWN_COUNTER
        global client_socket
        from models.sleep_detection import YAWN_COUNTER

        image = self.frame
        if type(image) != np.ndarray:
            _, image = TEMP_CAP.read()
        #self.create_Data()

        try:
            sleep_detect(image)
        except:
            TEMP_CAP.release()
            _, image = TEMP_CAP2.read()

        if YAWN_COUNTER == 5:
            client_socket.sendall("수고하셨습니다, 영상을 다시 재생합니다.".encode())
            YAWN_COUNTER = 6

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