import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models
from keras.layers import Dense, Activation

model=models.Sequential()
model.add(Dense(1024,input_shape=(10,)))
model.add(Activation('relu'))
model.add(Dense(2,activation='softmax'))

#NOTE: IMPORTANT NOTE
#Let's say you want to pass in a Numpy array into this network.
#Its shape must be (1,input_size), rather than the usual array shape (input_size,)
#This is because of Numpy's or Keras's bizarre input.

model=model.Sequential()
model.add(Dense(1024,input_shape=output_size_lstm)) #Fully connected layer
model.add(Activation('relu')) #Relu activation (assumed for now)
model.add(Dense(2,activation='softmax'))

#NOTE: The output will be a vector of the following format:
# [[number1,number2]], not the expected [number1,number2]
