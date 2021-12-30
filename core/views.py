from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist


FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 10
EYE_CLOSED_COUNTER = 0
BLINK_COUNT = 0
YAWN_COUNTER = 0
yawn_status = False 


faceDetector = dlib.get_frontal_face_detector() # 얼굴인식
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def add_overlays(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImage, 0)

        for face in faces:
            faceLandmarks = landmarkFinder(grayImage, face)
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

            # if ear < MINIMUM_EAR:
            #     EYE_CLOSED_COUNTER += 1
            # else:
            #     EYE_CLOSED_COUNTER = 0

            cv2.putText(image, "EAR: {}".format(round(ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
            cv2.putText(image, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 졸음 탐지 이벤트 발생

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



def eye_aspect_ratio(eye):
    
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

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