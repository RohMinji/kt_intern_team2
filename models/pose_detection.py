# 필요한 모듈 호출


import os
import math
import keras
import cv2
import numpy as np
import pandas as pd
from time import time
import time
import mediapipe as mp
import matplotlib.pyplot as plt

print("Success")
def detectPose(image, pose, display = True):
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize = [22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
#         # Also Plot the Pose landmarks in 3D.
#         mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks
    
    
def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.3)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode = False, min_detection_confidence=0.5)

# Initialize the VideoCapture object to read from the webcam.
#camera_video = cv2.VideoCapture(0)

# Initialize a resizable window.
#cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Curl counter variables
assigned_pose = "Lunge" # 이 부분 값 받는 걸로 바꿔주기
counter = 0

stage = None
label = "Stand"
model = keras.models.load_model('models/POSE_DETECTING_2021-12-31 11_41.h5') # 모델 잘 되는 걸로 변경필요

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
            
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            # Calculate the required angles.
            #----------------------------------------------------------------------------------------------------------------
            # Perform Pose landmark detection.
            frame, landmarks = detectPose(frame, pose_video, display = False)
            
            # Get the angle between the left shoulder, elbow and wrist points. 
            left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

            # Get the angle between the right shoulder, elbow and wrist points. 
            right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]) 

            # Get the angle between the left elbow, shoulder and hip points. 
            left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

            # Get the angle between the right hip, shoulder and elbow points. 
            right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

            # Get the angle between the left hip, knee and ankle points. 
            left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

            # Get the angle between the right hip, knee and ankle points 
            right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            
            target_names = ['Lunge', 'Squat', 'Stand', 'stretch'] ## 추가
            label = target_names[np.argmax(model.predict(pd.DataFrame([left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle]).T))]
            
            
            # Curl counter logic
            if label == "Stand":
                stage = "down"
                
                if counter == 5:
                    break
            
            if label == assigned_pose and stage =='down':
                time.sleep(0.2)
                counter += 1
                stage = "up"
                time.sleep(0.3)
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (235, 73), (211, 197, 203), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (159, 141, 106), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (159, 141, 106), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        if label == assigned_pose or label == "Stand":
            cv2.putText(image, label, (350, 60),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        else:
            cv2.putText(image, "UnKnown", (350, 60),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2) # B, G, R
            
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color = (211, 203, 197), thickness = 2, circle_radius = 2), 
                                mp_drawing.DrawingSpec(color = (159, 106, 141), thickness = 2, circle_radius = 2) 
                                 )                
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 