import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import cv2



import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#classes={0:'with mask',1:'without mask'}
new_model = keras.models.load_model('./best_facemask.h5')
#img = cv2.imread('./train/Mask/Mask10.jpg')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()
#img = cv2.resize(img,(150,150))
#img = np.reshape(img,[1,150,150,3])
#
#print(pred)
from skimage import transform

val_dir='./validate'

def load(img):
    np_img = Image.open(img)
    np_img = np.array(np_img).astype('float32')/255
    np_img = transform.resize(np_img,(150,150,3))
    np_img = np.expand_dims(np_img,axis=0)
    return np_img
img = load('./m2.jpg')
pred = new_model.predict(img)
print(pred)
if pred < 0.5:
    print('wearing mask')
else:
    print("not wearing mask")
