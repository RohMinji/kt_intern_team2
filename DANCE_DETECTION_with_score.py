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
from sklearn.preprocessing import Normalizer

# 코사인 유사도 구하는 코드
def findCosineSimilarity_1(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def compare_positions(trainer_video, user_video, keyp_list, dim = (420,720)):
    cap = cv2.VideoCapture(trainer_video)
    cam = cv2.VideoCapture(user_video) 
    cam.set(3, 646)
    cam.set(4, 364)
    score_list=[]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened() and cam.isOpened():
            ret_1, frame_1 = cam.read()
            ret_2, frame_2 = cap.read()
            
            if frame_1 is not None and frame_2 is not None:
                # Recolor Feed
                image1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
                image2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
                
                # Make Detections
                results1 = holistic.process(image1)
                results2 = holistic.process(image2)

                #Dancer keypoints and normalization
                transformer = Normalizer().fit(keyp_list)  
                keyp_list = transformer.transform(keyp_list)
                
                image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

                # Recolor image back to BGR for rendering
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                image1 = cv2.flip(image1, 1) # 좌우대칭
                
                output_image1 = image1.copy()
                imageRGB1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                
                results1 = holistic.process(image1)
                
                height, width, _ = image1.shape

                if results1.pose_landmarks:
                    points = results1.pose_landmarks.landmark
                    centers = {}

                    for i in range(len(points)):
                        x_axis = int(points[i].x * width + 0.5)
                        y_axis = int(points[i].y * width + 0.5)
                        center = [x_axis,y_axis]
                        centers[i] = center

                        k = -2
                        features = [0]*66
                        for j in range(0,33):
                            k += 2
                            try:
                                if k >= 66:
                                    break

                                features[k] = centers[j][0]
                                features[k + 1] = centers[j][1]
                            except:
                                features[k] = 0
                                features[k + 1] = 0
                        features = transformer.transform([features])

                        min_ = 100 # Intializing a value to get minimum cosine similarity score from the dancer array list with the user
                        for j in keyp_list:
                            sim_score = findCosineSimilarity_1(j,features[0])

                            #Getting the minimum Cosine Similarity Score
                            if min_ > sim_score:
                                min_ = sim_score
                                
                # Displaying the minimum cosine score
                cv2.putText(image1, str(min_), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # If the disctance is below the threshold
                if min_< 0.1:
                    cv2.putText(image1, "CORRECT STEPS", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    score_list.append(1)
                else:
                    cv2.putText(image1,  "NOT CORRECT STEPS", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    score_list.append(0)
                    
                cv2.imshow('User Window', image1)
                cv2.imshow('Dancer Window', image2)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cam.release()
        cap.release()
        cv2.destroyAllWindows()
        print(int(sum(score_list)/len(score_list)*100))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

keyp_list=[]
#f = open("C:/Users/User/Desktop/DANCE_DETECTION/keyp_list/waving_hands_keyplist.txt", 'r')
f = open("keyp_list/waving_hands_keyplist.txt", 'r')
while True: 
    line = f.readline()
    if not line: break
    line=line.replace("\n","")  
    keyp_list.append(list(map(str,line.split(" ")))[:-1])    
    
f.close()
#compare_positions(r'C:/Users/User/Desktop/DANCE_DETECTION/dance_video/sample_dance2.mp4', 0, keyp_list)
compare_positions(r'dance_video/sample_dance2.mp4', 0, keyp_list)

