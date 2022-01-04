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
from numpy import dot
from numpy.linalg import norm

avg_score="None"

# 코사인 유사도
def findCosineSimilarity_1(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def compare_positions(trainer_video, user_video, keyp_list, dim = (420,720)):
    cap = cv2.VideoCapture(trainer_video) # 댄스 영상
    cam = cv2.VideoCapture(user_video) # 실시간 웹캠
    cam.set(3, 646)
    cam.set(4, 364)
    fps_time = 0 
    
    key_ = 0
    tot_score=[]
    
    len_tot=1
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
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

                
                #Showing FPS
                cv2.putText(image2, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                #Displaying the dancer feed.
                image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
                cv2.imshow('Dancer Window', image2)
                
                # Recolor image back to BGR for rendering
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                image1 = cv2.flip(image1, 1) # 좌우대칭
                output_image1 = image1.copy()
                output_image2 = image2.copy()
                
                imageRGB1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                imageRGB2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                
                results1 = holistic.process(image1)
                results2 = holistic.process(image2)
                
                height, width, _ = image1.shape
                min_= -100 # Intializing a value to get minimum cosine similarity score from the dancer array list with the user
            
                if results1.pose_landmarks:
                    points = results1.pose_landmarks.landmark
                    features = []
                    for i in range(0, len(points)):
                        try:
                            features.append(int(points[i].x * width + 0.5))
                            features.append(int(points[i].y * width + 0.5))
                        except:
                            features.append(0)
                            features.append(0)

                            
                    sim_score = findCosineSimilarity_1(keyp_list[key_ * 33],features)
                    key_ += 1


                    # Displaying the minimum cosine score
                    cv2.putText(image1, str(sim_score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # If the disctance is below the threshold
                    if 0.98 <= sim_score <= 1:
#                         cv2.putText(image1, "CORRECT STEPS", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image1, "SCORE : " + str(int(sum(tot_score)/len_tot*100)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        tot_score.append(1)
                    else:
#                         cv2.putText(image1,  "NOT CORRECT STEPS", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image1, "SCORE : " + str(int(sum(tot_score)/len_tot*100)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        tot_score.append(0)
                    cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    len_tot=len(tot_score)

                
                # Render detections
                mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color = (211, 203, 197), thickness = 2, circle_radius = 2), 
                                        mp_drawing.DrawingSpec(color = (159, 106, 141), thickness = 2, circle_radius = 2) 
                                         )      
        
                # Display the user feed
                cv2.imshow('User Window', image1)

                fps_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cam.release()
        cap.release()
        cv2.destroyAllWindows()
        avg_score=(sum(tot_score) / len_tot) * 100
        print(avg_score)
#         print((sum(tot_score) / len(tot_score)) * 100)
        
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# txt 불러오기
temp = []
f = open(r"keyp_list/waving_hands_keyplist3.txt", 'r')

while True: 
    line = f.readline()
    if not line: break
    line = line.replace("\n","")  
    temp.append(list(map(str, line.split(" ")))[:-1])    
    
f.close()

keyp_list = []
for i in range(len(temp)):
    keyp_list.append(list(map(int, temp[i])))
    
# 함수 실행
compare_positions(r'dance_video/sample_dance2.mp4', 0, keyp_list)
