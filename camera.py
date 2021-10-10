from matplotlib.pyplot import imshow, winter
import numpy as np
from tensorflow import keras
import cv2
import os
from skimage import transform

new_model = keras.models.load_model('./best_facemask.h5')
font = cv2.FONT_HERSHEY_COMPLEX

class Video(object):
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def show(self):
        ret, frame = self.cap.read()
        height,width = frame.shape[:2]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gray = np.array(gray).astype('float32')/255
        gray = transform.resize(gray,(150,150,3))
        gray = np.expand_dims(gray,axis=0)
        pred = new_model.predict(gray)
        print(pred)
        if pred < 0.1:
            cv2.putText(frame,'wearing mask',(10,height-20),font,1,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,'not wearing mask',(10,height-20),font,1,(255,255,255),1,cv2.LINE_AA)
        ret,jpg = cv2.imencode('.jpg',frame)
        return jpg.tobytes() 
