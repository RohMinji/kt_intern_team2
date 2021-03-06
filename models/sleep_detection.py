import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
import datetime
import pandas as pd
from django.http import JsonResponse
import mediapipe as mp
import random
import json
import time

from models.pose_detection import pose_detect
from models.dance_detection import compare_positions

MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 3
EYE_CLOSED_COUNTER=0
BLINK_COUNT=0
YAWN_COUNTER = 0
YAWN_STATUS = False
STAGE = 0
sy_exist = 0
videoValue = 0

# Load Models for Detecting Face
detector = dlib.get_frontal_face_detector()

# Load Models for Extracting Landmarks of Face
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat') #랜드마크 추출
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Call Module for Media Pipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load Vector File of Train Data 
temp = []
f = open('models/ppap_keyplist.txt', 'r')

while True: 
    line = f.readline()
    if not line: break
    line = line.replace("\n","")  
    temp.append(list(map(str, line.split(" ")))[:-1])    
    
f.close()

keyp_list = []
for i in range(len(temp)):
    keyp_list.append(list(map(int, temp[i])))


# Dictionary of Eye Size
eyesize_dic={}
# Dictionary of User's Vacancy
empty_dic={}

# Call Pose Detection Function
def call_pose_func(assigned_pose):
    try:
        cap = cv2.VideoCapture(0)
        pose_detect(cap, assigned_pose)
        cap.release()
        cv2.destroyAllWindows()
        return 0
    except:
        cv2.VideoCapture(0).release()
        cv2.destroyAllWindows()
        return 0

# Call Dance Detection Function
def call_dance_func():
    global YAWN_COUNTER
    # from core.views import client_socket

    try:
        dance_cap = cv2.VideoCapture(0)
        compare_positions('static/video/dance_video.mp4', dance_cap, keyp_list)
        dance_cap.release()
        cv2.destroyAllWindows()
        # Call GiGA-Genie
        # client_socket.sendall("수고하셨습니다, 영상을 다시 재생합니다.".encode())
        return 0
    except:
        cv2.VideoCapture(0).release()
        cv2.destroyAllWindows()
        return 0

# Sleep Detection Function
def sleep_detect(image):
    global YAWN_STATUS
    global grayImage
    global BLINK_COUNT
    global YAWN_COUNTER
    global EYE_CLOSED_COUNTER
    # global client_socket
    global STAGE
    global sy_exist
    global videoValue

    # from core.views import client_socket

    # Collect Current Time
    now = datetime.datetime.now()
    now = now.strftime("%H:%M:%S")
    now = str(now)

    # Data Preprocessing
    image_landmarks, lip_distance = mouth_open(image)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(grayImage, 0)
    sy_exist = len(faces)

    # Detect User's Vacancy     
    if len(faces) < 1:
        empty_dic[now] = 1
        eyesize_dic[now] = 0
        cv2.putText(image, "No Student", (50,450), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 2)
    else:
        pass

    for face in faces:
        ear= calEAR(face, image)
        eyesize_dic[now] = 1 / ear

        # Count Duration of User's Blink
        if ear < MINIMUM_EAR:
            EYE_CLOSED_COUNTER += 1
        else:
            EYE_CLOSED_COUNTER = 0

        # Detect User's Drowsiness
        if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
            cv2.putText(image, "Sleep", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
            BLINK_COUNT += 1
            cv2.putText(image, "Count: {}".format(int((BLINK_COUNT) / 5)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
    
    # Detect User's Yawn
    prev_yawn_status = YAWN_STATUS
    if lip_distance > 25:
        YAWN_STATUS = True 
    
        output_text = " Yawn Count: " + str(YAWN_COUNTER + 1)
        cv2.putText(image, output_text, (50, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,127), 2)
    else:
        YAWN_STATUS = False 

    ############ Reflect Stage of Drowsiness        
    if prev_yawn_status == True and YAWN_STATUS == False:
        YAWN_COUNTER += 1
        if YAWN_COUNTER == 3 and STAGE == 0:
            assigned_pose = random.choice(["Squat", "Lunge"]) # select pose
            # if assigned_pose == "Squat":
            #     client_socket.sendall(("졸음 깨기 1단계를 시작합니다. 스쿼트를 3. 회 시행해주세요.").encode())
            # elif assigned_pose == "Lunge":
            #     client_socket.sendall(("졸음 깨기 1단계를 시작합니다. 런지를 3. 회 시행해주세요.").encode())
            STAGE = 1
            videoValue += 1
            time.sleep(1)
            call_pose_func(assigned_pose)
            videoValue = 0



        elif YAWN_COUNTER == 5 and STAGE == 1:
            # client_socket.sendall("졸음 깨기 2단계를 시작합니다. 음악에 맞춰 춤을 춰주세요.".encode())
            STAGE = 2
            # client_socket.sendall("123".encode())
            videoValue += 1
            call_dance_func()
            videoValue = 0

    elif int((BLINK_COUNT) / 5) == 3 and STAGE == 0:
        assigned_pose = random.choice(["Squat", "Lunge"]) # select pose
        # if assigned_pose == "Squat":
        #     client_socket.sendall(("졸음 깨기 1단계를 시작합니다. 스쿼트를 3. 회 시행해주세요.").encode())
        # elif assigned_pose == "Lunge":
        #     client_socket.sendall(("졸음 깨기 1단계를 시작합니다. 런지를 3. 회 시행해주세요.").encode())
        STAGE = 1
        videoValue = 1
        time.sleep(1)
        call_pose_func(assigned_pose)
        videoValue = 0
        
    elif int((BLINK_COUNT) / 5) == 6 and STAGE == 1:
        videoValue = 1
        # client_socket.sendall("졸음 깨기 2단계를 시작합니다. 음악에 맞춰 춤을 춰주세요.".encode())
        STAGE = 2
        # client_socket.sendall("123".encode())
        call_dance_func()
        videoValue = 0

# Get Landmarks
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# Get EAR of Each Eye
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

# Return Image with Landmarks
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

# Get Points of top lip
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

# Get Points of bottom lip
def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

# Return Lip Distance 
def mouth_open(imagee):
    landmarks = get_landmarks(imagee)
    
    if landmarks == "error":
        return imagee, 0
    
    image_with_landmarks = annotate_landmarks(imagee, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

# Return Mean Value of Left & Right eye's EAR
def calEAR(face, image):
    faceLandmarks = predictor(grayImage, face)
    faceLandmarks = face_utils.shape_to_np(faceLandmarks)

    leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
    rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)

    cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
    cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)
    
    return ear

# Course Video Stop Control 
def videoStop(request):
    global videoValue
    global sy_exist
    data = {
        "videoValue": videoValue,
        "sy_exist" : sy_exist,
    }
    return JsonResponse(data)