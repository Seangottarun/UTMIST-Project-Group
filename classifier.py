%load_ext tensorboard
import tensorflow as tf
import numpy as np

class Classifier:
    def __init__(self, config):
        self.config = config
        self.output_size = self.config.object_labels
        self.input_size = len(self.config.LSTM_output)
        self.hidden_size = self.config.hidden_size
    
    def __call__(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = self.input_size),
            tf.keras.layers.Dense(self.hidden_size, activation = "relu"),
            tf.keras.layers.Dense(self.output_size, activation="softmax")
        ])


MyClassifier = Classifier()
model = MyClassifier()
model.compile(optimizer = 'Adam',
loss = "sparse_categorical_crossentropy",
metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

%tensorboard --logdir logs/fit