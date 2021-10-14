import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

train_dir = './Train'
val_dir='./Validation'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
  )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validaton_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

conv_base = InceptionResNetV2(weights= 'imagenet',
                  include_top= False,
                  input_shape=(150,150,3))

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False

model.summary()
model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])
checkpoint_c1= keras.callbacks.ModelCheckpoint("best_facemask.h5",save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validaton_generator,
    validation_steps=20,
    callbacks = [checkpoint_c1]
)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.save("fm.h5")
