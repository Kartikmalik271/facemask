import numpy as np
from tensorflow import keras
import cv2
from skimage import transform

new_model = keras.models.load_model('./fm.h5')
font = cv2.FONT_HERSHEY_COMPLEX

class Video(object):
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def show(self):
        ret, frame = self.cap.read()
        height,width = frame.shape[:2]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = np.array(gray).astype('float32')/255
        gray = transform.resize(gray,(150,150,3))
        gray = np.expand_dims(gray,axis=0)
        pred = new_model.predict(gray)
        if pred < 0.01:
            cv2.putText(frame,'wearing mask',(15,height-30),font,1,(0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(frame,'not wearing mask',(15,height-30),font,1,(0,0,255),2,cv2.LINE_AA)
        ret,jpg = cv2.imencode('.jpg',frame)
        return jpg.tobytes() 
