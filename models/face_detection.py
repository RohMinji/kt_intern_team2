import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from django.http import JsonResponse

# Loading User's model
model = load_model('models/FACE_MODEL.h5', compile=False)
model.summary()

# Counting the number of frames to identify user
SY_COUNT = 0

# Face Detection
def face_detect(cam, image):
    global SY_COUNT
    webcam = cam
    web_frame = image

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    
    # Loop through Frames
    img = cv2.resize(web_frame, (224, 224), interpolation = cv2.INTER_AREA)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0]) # predicted class 0, 1, 2

    if predicted_class == 0:
        SY_COUNT += 1

# Deliver Data
def sy_detection(request):
    global SY_COUNT
    data = {
        "SY_COUNT": SY_COUNT,
    }
    return JsonResponse(data)