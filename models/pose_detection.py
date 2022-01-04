import math
import keras
import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
import matplotlib.pyplot as plt
import random
from joblib import load

# Initializing mediapipe pose class
mp_pose = mp.solutions.pose

# Initializing mediapipe drawing class, useful for annotation
mp_drawing = mp.solutions.drawing_utils

# Setup Pose function for video
pose_video = mp_pose.Pose(static_image_mode = False, min_detection_confidence=0.5)

# Curl counter variables
model = load('models/POSE_MODEL.joblib')

counter = 0
stage = None
label = "Stand"

# Detect User's Pose
def detectPose(image, pose, display = True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    
    # Check if any landmarks are detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    if display:
        plt.figure(figsize = [22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    else:
        return output_image, landmarks
    
# Calculate Angle    
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
    return angle


# Evaluate User's Motion
def pose_detect(cap, assigned_pose):
    print("POSE DETECT FUNCTION START")
    global counter
    global stage
    global label

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        Time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image = cv2.flip(image, 1)
            
            # Make detection
            results = pose.process(image)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Calculate the required angles
            try:
                # Perform Pose landmark detection
                frame, landmarks = detectPose(frame, pose_video, display = False)
                
                # Get the angle between the left shoulder, elbow and wrist points
                left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

                # Get the angle between the right shoulder, elbow and wrist points
                right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]) 

                # Get the angle between the left elbow, shoulder and hip points
                left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

                # Get the angle between the right hip, shoulder and elbow points
                right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

                # Get the angle between the left hip, knee and ankle points
                left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

                # Get the angle between the right hip, knee and ankle points 
                right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                
                target_names = ['Lunge', 'Squat', 'Stand', 'stretch']
                label = target_names[np.argmax(model.predict(pd.DataFrame([left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle]).T))]
                
                if Time + 5 < time.time():
                    if label == "Stand":
                        if counter == 3:
                            cap.release()
                            cv2.destroyAllWindows()
                            break
                        else:
                            stage = "down"

                    if label == assigned_pose and stage =='down':
                        time.sleep(0.2)
                        counter += 1
                        stage = "up"
                        time.sleep(0.3)
                else:
                    pass
            except:
                pass
            
            # Display Status
            cv2.rectangle(image, (0, 0), (235, 73), (211, 197, 203), -1)
            
            # Rep data
            cv2.putText(image, 'COUNT', (15, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (159, 141, 106), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (159, 141, 106), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            if label == assigned_pose or label == "Stand":
                cv2.putText(image, label, (350, 60),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)
            else:
                cv2.putText(image, "UnKnown", (350, 60),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3) # B, G, R
                
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