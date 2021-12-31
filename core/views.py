from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import dlib
import numpy as np
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist


FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
 


faceDetector = dlib.get_frontal_face_detector() # 얼굴인식
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 3
EYE_CLOSED_COUNTER=0
BLINK_COUNT=0
YAWN_COUNTER = 0
yawn_status = False

class VideoCamera(object):
    


    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()


    def add_overlays(self, image):
        global EYE_CLOSED_COUNTER 
        global BLINK_COUNT
        global YAWN_COUNTER
        global yawn_status

        image_landmarks, lip_distance = VideoCamera.mouth_open(self.frame)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImage, 0)

        # 학생 자리 비움 인식
        if len(faces)<1:
            cv2.putText(self.frame, "No Student", (50,450),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

        for face in faces:
            faceLandmarks = landmarkFinder(grayImage, face)
            faceLandmarks = face_utils.shape_to_np(faceLandmarks)

            leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
            rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

            leftEAR = VideoCamera.eye_aspect_ratio(leftEye)
            rightEAR = VideoCamera.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(self.frame, [leftEyeHull], -1, (255, 0, 0), 2)
            cv2.drawContours(self.frame, [rightEyeHull], -1, (255, 0, 0), 2)
            if ear < MINIMUM_EAR:
                EYE_CLOSED_COUNTER += 1
                print(EYE_CLOSED_COUNTER)
            else:
                print("counter 0")
                EYE_CLOSED_COUNTER = 0
                #cv2.putText(frame, "EAR: {}".format(round(ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                    cv2.putText(self.frame, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    BLINK_COUNT += 1
                    cv2.putText(self.frame, "Count: {}".format(int((BLINK_COUNT)/5)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #하품 인식
        prev_yawn_status = yawn_status
        if lip_distance>25:
            
            yawn_status = True 
            #cv2.putText(frame, "Subject is Yawning", (50,450), 
            #           cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            
            output_text = " Yawn Count: " + str(YAWN_COUNTER + 1)

            cv2.putText(self.frame, output_text, (50,50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        else:
            yawn_status = False 
            
        if prev_yawn_status == True and yawn_status == False:
            YAWN_COUNTER += 1

        #cv2.imshow('Live Landmarks', image_landmarks )
        #cv2.imshow('Yawn Detection', self.frame )

    def eye_aspect_ratio(eye):
    
        p2_minus_p6 = dist.euclidean(eye[1], eye[5])
        p3_minus_p5 = dist.euclidean(eye[2], eye[4])
        p1_minus_p4 = dist.euclidean(eye[0], eye[3])
        ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear

    # Landmark 가져오기
    def get_landmarks(im):
        rects = faceDetector(im, 1)

        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        return np.matrix([[p.x, p.y] for p in landmarkFinder(im, rects[0]).parts()])

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
        landmarks = VideoCamera.get_landmarks(image)
        
        if landmarks == "error":
            return image, 0
        
        image_with_landmarks = VideoCamera.annotate_landmarks(image, landmarks)
        top_lip_center = VideoCamera.top_lip(landmarks)
        bottom_lip_center = VideoCamera.bottom_lip(landmarks)
        lip_distance = abs(top_lip_center - bottom_lip_center)
        return image_with_landmarks, lip_distance

        #cv2.imshow('Result', image_with_landmarks)
        #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def get_frame(self):
        image = self.frame
        self.add_overlays(image)
        # jpeg encoding
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



# while True:
#     # (status, image) = webcamFeed.read()
    

#     cv2.imshow("Frame", image)
#     if cv2.waitKey(1) == 13: #13 is the Enter Key
#         break

def index(request):
    return render(request, "core/index.html")

@gzip.gzip_page
def cam_test(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("Error")
        pass