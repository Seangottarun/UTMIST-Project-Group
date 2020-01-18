# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np

#variables as input: Outsize:(output size of the glimpse vector), x_coord, y_coord from the glimpse that we obtain as input

# MNIST dataset imported
offset1=[]
for i in range(0,batch_size,1):
    offset1+=[[x_coordinate,y_coordinate]]
output = tf.image.extract_glimpse(image, 54, 54, offset=offset1, centered=False)

#First Part: Convnet
#Fully Connected Layer

model=model.Sequential()
model.add(Conv2D(64,kernel_size=5,activation='relu',input_shape=(glimpse_height,glimpse_width,channels)))
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(Conv2D(128,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(OutSize,activation='softmax'))

#Endshape: EMPTY x 1

#Second Part:
loc=loc.Sequential()
loc.add(Dense(1024, input_shape=(2,1)))
loc.add(Dense(OutSize, activation="softmax"))

#Endshape: EMPTY X 1

final = tf.math.multiply(loc, model)

final.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
