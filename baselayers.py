import tensorflow as tf

class Layers(object):
    def __init__(self):
        pass

    def conv(self, inputs, filters, kernel_size, strides, padding='SAME', name='conv_layer'):
        """
        Defines a convolutional layer.

        Variables:
        inputs: tensor of shape [batch, in_height, in_width, in_channels]
        filters: tensor of shape [filter_height, filter_width, in_channels, out_channels]
        kernel_size, strides, padding: 1-D scalars for corresponding variables.
        
        Returns tensor of shape [batch, out_height, out_width, out_channels]
        """
        input_channels = inputs[-1]
        kernel = tf.Variable(tf.random.truncated_normal(shape=[kernel_size, kernel_size, input_channels, filters]),
                             dtype=tf.float32, name='kernel')
        bias = tf.Variable(tf.zeros(shape=[filters]), name='bias')
        conv = tf.nn.conv2d(inputs, filter=kernel,
                            strides=[1, strides, strides, 1],
                            padding=padding, name='conv')
        out = tf.nn.relu(conv + bias, name='relu')
        return out
        
    def max_pool(self, inputs, kernel_size, strides, padding='VALID', name='maxpool_layer'):
        """
        Defines a maxpooling layer. 

        Variables:
        inputs: tensor of shape [batch, in_height, in_width, channels]
        kernel_size, strides: 1-D scalars for corresponding variable

        Returns tensor of shape [batch, out_height, out_width, channels]
        """
        pool = tf.nn.max_pool2d(inputs, ksize=[1, ksize, ksize, 1],
                                strides=[1, strides, strides, 1], 
                                padding=padding, name=name)
        return pool

    def fc(self, inputs, out_dim, name="fc_layer"):
        """
        Defines a fully-connected layer.
        
        Variables:
        inputs: flattened tensor of shape [batch, height * width * channels]
        out_dim: 1-D scalar for new output dimension

        Returns tensor of shape [batch, out_dim]. Note that NO REGULARIZATION HAS BEEN DONE.
        """
        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initializer(shape=[in_dim, out_dim]), name='weights')
        b = tf.Variable(tf.zeros(shape=[out_dim]), name='bias')
        out = tf.matmul(inputs, w) + b
        return out
