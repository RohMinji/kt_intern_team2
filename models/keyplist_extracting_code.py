#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# 필요한 모듈 호출

import os
import math
import cv2
import numpy as np
import pandas as pd
from time import time
import time
import mediapipe as mp
import matplotlib.pyplot as plt

import mediapipe as mp
import cv2
from sklearn.preprocessing import Normalizer


def dance_video_processing(video_path = r'dance_video/sample_dance2.mp4',showBG = True):
    cap = cv2.VideoCapture(r"dance_video/sample_dance2.mp4")

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    keypoints_list = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        fps_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if frame is not None:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_image = image.copy()
                
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                height, width, _ = image.shape

                points = results.pose_landmarks.landmark
                centers = {}

                for i in range(len(points)):
                    x_axis = int(points[i].x * width + 0.5)
                    y_axis = int(points[i].y * width + 0.5)
                    center=[x_axis,y_axis]
                    centers[i] = center
                    keypoints_list.append(centers)
                             

                # To display fps
                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color = (211, 203, 197), thickness = 2, circle_radius = 2), 
                                mp_drawing.DrawingSpec(color = (159, 106, 141), thickness = 2, circle_radius = 2) 
                                 )  
        
                cv2.imshow('Dancer', image)
                fps_time = time.time()

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
    return keypoints_list


def get_position(video_path= r'dance_video/sample_dance2.mp4',showBG = True):
    keypoints_list = dance_video_processing()
    keyp_list = []
    
    for i in range(0, len(keypoints_list)):
        features = []
        for j in range(0, len(keypoints_list[i])):
            try:
                features.append(keypoints_list[i][j][0])
                features.append(keypoints_list[i][j][1])
                
            except:
                features.append(0)
                features.append(0)
                
        keyp_list.append(features)
        
    return keyp_list, keypoints_list


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

keyp_list,keypoints_list = get_position()


# writedata.py
f = open("keyp_list/waving_hands_keyplist3.txt", 'w')

for lst1 in keyp_list:
    for lst2 in lst1:
        f.write(str(lst2))
        f.write(" ")
    f.write("\n")
    
f.close()

