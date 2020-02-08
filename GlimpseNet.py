# TensorFlow and tf.keras
%load_ext tensorboard
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

#variables as input: Outsize:(output size of the glimpse vector), x_coord, y_coord from the glimpse that we obtain as input
class GlimpseNet(tf.keras.Model):
    def __init__(self, config1):
        self.config = config1
        #First Part: Convnet
        #Fully Connected Layer
        
        #define inputs
        glimpse = Input(shape=(self.config.height, self.config.width, self.config.batch_size))
        location = Input(shape=(2, self.config.batch_size))

        #gets a patch of the image as an output vector
        #3 hidden 2D convolutional layers
        conv_1 = Conv2D(64,kernel_size=5,activation='relu',input_shape=(self.config.height,self.config.width,self.config.color_channels))(glimpse)
        conv_2 = Conv2D(64,kernel_size=3,activation='relu')(conv_1)
        conv_3 = Conv2D(128,kernel_size=3,activation='relu')(conv_2)
        flatten = Flatten()(conv_3)

        #FC layer
        fc_1 = Dense(1024,activation='relu')(flatten)
        fc_2 = Dense(self.config.OutSize,activation='softmax')(fc_1)

        #Endshape: EMPTY x 1

        #Second Part:
        #gets the location tuple
        loc_fc_1 = Dense(1024, input_shape=(1,2))(location)
        loc_fc_2 = Dense(self.config.OutSize, activation="softmax")(loc_fc_1)

        # combining the location and image output vectors to create a final glimpse vector
        final = tf.math.multiply(fc_2, loc_fc_2)

    def __call__(self, image, location_tuple):
        offset1=[]
        for i in range(0,self.config.batch_size,1):
            offset1+=[[location_tuple[0], location_tuple[1]]]
        output = tf.image.extract_glimpse(image, self.config.height, self.config.width, offset=offset1, centered=False)

        model = Model(inputs=[output, location_tuple], outputs=final)
        return final
