# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:27:05 2021

@author: User
"""

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import imutils
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import csv

 


#PREDICTOR_PATH = "C:/Users/User/Downloads/shape_predictor_68_face_landmarks.dat"
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
predictor = dlib.shape_predictor('C:/Users/User/Downloads/shape_predictor_68_face_landmarks.dat') #랜드마크 추출
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0) #비디오 불러오기
#변수 설정
#눈 깜빡임
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 5
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
EYE_CLOSED_COUNTER = 0
BLINK_COUNT = 0

#하품
YAWN_COUNTER = 0
yawn_status = False 

# Landmark 가져오기
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def eye_aspect_ratio(eye):
    
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def calEAR(face):
    
    faceLandmarks = predictor(grayframe, face)
    faceLandmarks = face_utils.shape_to_np(faceLandmarks)

    leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
    rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)

    cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 2)
    cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 2)
    
    return ear


lip_list = []
ear_list = []
out_list = []

while True:
    
    ret, frame = cap.read()   
    image_landmarks, lip_distance = mouth_open(frame)
    
    frame = imutils.resize(frame, width=800)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #ear 눈 감고 지속되는 시간 측정, 초과시 경고 알림
    faces = detector(grayframe, 0)
    
    # 학생 자리 비움 인식
    if len(faces)<1:
        out_list += [1]
        cv2.putText(frame, "No Student", (50,450),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
    else:
        out_list += [0]
    # 사람 단위로 
    for face in faces:
        ear= calEAR(face)
        if ear < MINIMUM_EAR:
            ear_list += [ear]
            EYE_CLOSED_COUNTER += 1
           # print(EYE_CLOSED_COUNTER)
        else:
            ear_list += [0]
            EYE_CLOSED_COUNTER = 0
            #cv2.putText(frame, "EAR: {}".format(round(ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
            cv2.putText(frame, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            BLINK_COUNT += 1
            cv2.putText(frame, "Count: {}".format(int((BLINK_COUNT)/5)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #하품 인식
    prev_yawn_status = yawn_status
    if lip_distance>25:
        
        lip_list = lip_list + [lip_distance]
        
        yawn_status = True 
        #cv2.putText(frame, "Subject is Yawning", (50,450), 
        #           cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        
        output_text = " Yawn Count: " + str(YAWN_COUNTER + 1)

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    else:
        lip_list = lip_list+ [0]
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        YAWN_COUNTER += 1

    #cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
    
        #csv파일로 추출
        max1 = int(max(lip_list))
        if max1 != 0:
            for i in range(len(lip_list)):
                lip_list[i] = int(lip_list[i])/max1
        max2 = max(ear_list)
        if max2 != 0:
            for i in range(len(ear_list)):
                ear_list[i] = ear_list[i]/max2
        dataframe1 = pd.DataFrame([lip_list, ear_list, out_list])
        dataframe = dataframe1.transpose()
        dataframe.index.name="fps"
        dataframe.to_csv("RA.csv", header = True, index = True)
        
        # 분석 결과 시각화
        df = pd.read_csv("C:/Users/User/.spyder-py3/RA.csv")
        fpsList  = df ['fps'].tolist()
        lipData   = df ['0'].tolist()
        earData   = df ['1'].tolist()
        moveData = df ['2'].tolist()

        plt.plot(fpsList, lipData,  label = 'Lip Data', linewidth=1)
        plt.plot(fpsList, earData,  label = 'Ear Data', linewidth=1)
        plt.plot(fpsList, moveData, label = 'Move Data', linewidth=1)


        plt.xlabel('fps')
        plt.ylabel('Data units in number')
        plt.legend(loc='upper left')
        #plt.xticks(range(fpsList,5))
        #plt.xticks(fpsList)
        plt.yticks(np.arange(0,1, 0.1))
        plt.title('Result')
        plt.show()
        
        break
        
cap.release()
cv2.destroyAllWindows() 