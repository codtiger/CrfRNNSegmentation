import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class UpSampleLayer(Layer):

    def __init__(self, kernel_size, out_shape, class_num, stride=2, **kwargs):
        self.kernel_size = kernel_size
        self.out_shape = out_shape
        self.class_num = class_num
        self.stride = stride
        super(UpSampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bilinear_filter = self.get_deconv_filter(self.kernel_size, self.class_num)
        super(UpSampleLayer, self).build(input_shape)

    def call(self, inputs):
        # bilinear_filter = self.get_deconv_filter(self.kernel_size, self.class_num)
        batch_size = tf.shape(inputs)[0]
        self.out_shape = [batch_size, self.out_shape, self.out_shape, 21]
        if self.trainable:
            transpose_kernel = tf.get_variable(self.name
                                               , initializer=self.bilinear_filter, trainable=True)
        else:
            transpose_kernel = tf.get_variable(self.name
                                               , initializer=self.bilinear_filter, trainable=False)
        output = tf.nn.conv2d_transpose(inputs, transpose_kernel, output_shape=tf.stack(self.out_shape)
                                        , strides=[1, self.stride, self.stride, 1], padding='SAME')

        return output

    def compute_output_shape(self, input_shape):
        self.out_shape[0] = input_shape[0]
        self.out_shape = tuple(self.out_shape)
        return self.out_shape

    def get_deconv_filter(self, size, out_size):
        factor = size + 1 // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        x_filter, y_filter = np.ogrid[:size, :size]
        weights = (1 - np.abs(x_filter - center) / factor) * \
                  (1 - np.abs(y_filter - center) / factor)
        bilinear_filter = np.zeros((size, size, out_size, out_size))
        for i in range(out_size):
            bilinear_filter[:, :, i, i] = weights
        return bilinear_filter.astype(np.float32)

