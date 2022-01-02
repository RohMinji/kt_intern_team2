import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
import subprocess
import mediapipe as mp

# from core.views import pose_detection
# from models import pose_detection
# from models.pose_detection import detectPose

MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 3
EYE_CLOSED_COUNTER=0
BLINK_COUNT=0
YAWN_COUNTER = 0
YAWN_STATUS = False

detector = dlib.get_frontal_face_detector() # 얼굴인식
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #랜드마크 추출
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def sleep_detect(image):
    global YAWN_STATUS
    global grayImage
    global BLINK_COUNT
    global YAWN_COUNTER
    global EYE_CLOSED_COUNTER

    image_landmarks, lip_distance = mouth_open(image)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(grayImage, 0)

    if len(faces)<1:
        cv2.putText(image, "No Student", (50,450),
                cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

    for face in faces:
        ear= calEAR(face, image)
        if ear < MINIMUM_EAR:
            EYE_CLOSED_COUNTER += 1
        else:
            EYE_CLOSED_COUNTER = 0
        #cv2.putText(frame, "EAR: {}".format(round(ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
            cv2.putText(image, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            BLINK_COUNT += 1
            cv2.putText(image, "Count: {}".format(int((BLINK_COUNT)/5)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    prev_yawn_status = YAWN_STATUS
    if lip_distance>25:
        YAWN_STATUS = True 
    #cv2.putText(frame, "Subject is Yawning", (50,450), 
    #           cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
    
        output_text = " Yawn Count: " + str(YAWN_COUNTER + 1)

        cv2.putText(image, output_text, (50,50),
                cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    else:
        YAWN_STATUS = False 
        
    if prev_yawn_status == True and YAWN_STATUS == False:
        YAWN_COUNTER += 1
        if YAWN_COUNTER >= 3:
            # subprocess.call("POSE_DETECT_final.py", shell=True)
            # cv2.VideoCapture(0).release()
            try:
                cv2.VideoCapture(0).release()
                subprocess.call(['python.exe', "models/pose_detection.py"], shell=True)
                return
            except:
                print("Error to end the video")
            # mp_pose = mp.solutions.pose
            # pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.3)

            # pose_detection.detectPose(image, pose)

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


def mouth_open(imagee):
    landmarks = get_landmarks(imagee)
    
    if landmarks == "error":
        return imagee, 0
    
    image_with_landmarks = annotate_landmarks(imagee, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


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

