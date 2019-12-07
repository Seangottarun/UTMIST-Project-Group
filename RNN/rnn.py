from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_datasets as tfds

import tensorflow as tf

class RecurrentNet(tf.keras.Model):

  def __init__(self, config, **kwargs):
    super(RecurrentNet, self).__init__(**kwargs)

    self.config = config

    self.batch_size = config["batch_size"]
    self.input_dim = config["input_dim"]
    self.units = config["units"]
    self.output_size = config["output_size"]

    self.lstm_layer = tf.keras.layers.LSTM(self.units, input_shape=(None, self.input_dim))
    self.batch_norm = layers.BatchNormalization()
    self.classifier = layers.Dense(self.output_size, activation='softmax')


  def call(self, inputs):
    x = self.lstm_layer(inputs)
    x = self.batch_norm(x)
    return self.classifier(x)
