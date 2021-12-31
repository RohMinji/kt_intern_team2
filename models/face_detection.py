import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('models/keras_model.h5', compile=False)
model.summary()

print("hello ")

def face_detect(cam, image):
    webcam = cam
    web_frame = image

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    
    # loop through frames
    img = cv2.resize(web_frame, (224, 224), interpolation = cv2.INTER_AREA)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0]) # 예측된 클래스 0, 1, 2
    
    if predicted_class == 0:
        me = "안녕하세요 승용님, 학습을 시작하겠습니다."
    elif predicted_class == 1:
        me = "교육생이 아닙니다."        
    elif predicted_class == 2:
        me = ""
    print("predicted_class", predicted_class)
