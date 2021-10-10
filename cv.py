from matplotlib.pyplot import winter
import numpy as np
from tensorflow import keras
import cv2
import os
from skimage import transform


new_model = keras.models.load_model('./best_facemask.h5')
cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
count = 0
score = 0
thicc = 2

while(True):
    ret, frame = cap.read()
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
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
