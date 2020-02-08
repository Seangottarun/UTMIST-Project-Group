# TensorFlow and tf.keras
%load_ext tensorboard
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
!rm -rf ./logs/

#variables as input: Outsize:(output size of the glimpse vector), x_coord, y_coord from the glimpse that we obtain as input
class GlimpseNet(tf.keras.Model):
    def __init__(self, config1):
        self.config = config1


    def __call__(self,location_tuple):
        #HEYYY: I think we need to add the image as an input to the __call__ function.
        offset1=[]
        for i in range(0,self.config.batch_size,1):
            offset1+=[[location_tuple[0],location_tuple[1]]]
        output = tf.image.extract_glimpse(image, config.height, config.width, offset=offset1, centered=False)

        #First Part: Convnet
        #Fully Connected Layer

        #gets a patch of the image as an output vector
        model=model.Sequential()
        #3 hidden 2D convolutional layers
        model.add(Conv2D(64,kernel_size=5,activation='relu',input_shape=(self.config.height,self.config.width,self.config.color_channels)))
        model.add(Conv2D(64,kernel_size=3,activation='relu'))
        model.add(Conv2D(128,kernel_size=3,activation='relu'))
        model.add(Flatten())
        #FC layer
        model.add(Dense(1024,activation='relu'))
        model.add(Dense(self.config.OutSize,activation='softmax'))

        #Endshape: EMPTY x 1

        #Second Part:
        #gets the location tuple
        loc=loc.Sequential()
        loc.add(Dense(1024, input_shape=(1,2)))
        loc.add(Dense(self.config.OutSize, activation="softmax"))

        #model/loc compilation
        model.compile(optimizer='adam',loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loc.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        #Endshape: EMPTY X 1
        model_test=model.predict(#random vec)
        loc_test=loc.predict(#random vec)

        #combining the location and image output vectors to create a final glimpse vector
        final = tf.math.multiply(model_test, loc_test)

        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return final
