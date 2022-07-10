import os
import time
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf
import high_dim_filter_loader
import tensorflow.keras.backend as K
custom_module = high_dim_filter_loader.custom_module


def random_uniform(size):

    return np.random.uniform(0, 1, size)


def random_normal(size):

    return np.random.randn(*size)


class CrfRnn(Layer):

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):

        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta

        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        super(CrfRnn, self).__init__(**kwargs)


    def _comp_mat_initializer(self, input_shape):

        return -1 * np.eye(input_shape[0], dtype=np.float32)

    def build(self, input_shape):

        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=random_normal)
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer=random_normal)
        self.compatablity_matrix = self.add_weight(name='compatablity_matrix',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=self._comp_mat_initializer)
        super(CrfRnn, self).build(input_shape)

    def call(self, inputs):

        unaries = tf.transpose(inputs[0][:, :, :, :], perm=[0, 3, 1, 2])
        rgb = tf.transpose(inputs[1][:, :, :, :], perm=[0, 3, 1, 2])

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        spatial_norm_values = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                            theta_gamma=self.theta_gamma)

        bilateral_norm_values = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                              theta_alpha=self.theta_alpha,
                                                              theta_beta=self.theta_beta)
        q_values = unaries


        for i in range(self.num_iterations):
            batch_size = K.shape(inputs[0])[0]
            normalized = tf.nn.softmax(q_values, axis=0)

            spatial_out = custom_module.high_dim_filter(normalized, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            bilateral_out = custom_module.high_dim_filter(normalized, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)

            spatial_out = tf.div(spatial_out, spatial_norm_values)
            bilateral_out = bilateral_out / bilateral_norm_values

            message_passing = tf.add(tf.matmul(self.spatial_ker_weights,
                                               tf.reshape(spatial_out, [self.num_classes,
                                                                        -1])),
                                     tf.matmul(self.bilateral_ker_weights,
                                               tf.reshape(spatial_out, (self.num_classes
                                                                        , -1))))
            print (message_passing.shape)
            pairwise = tf.matmul(self.compatablity_matrix, message_passing)

            pairwise = tf.reshape(pairwise, (batch_size, c, h, w))

            q_values = unaries - pairwise

        # TODO:maybe add a softmax layer after this for future optimization

        return tf.transpose(tf.reshape(q_values, (batch_size, c, h, w)), (0, 3, 2, 1))

    def compute_output_shape(self, input_shape):

        return input_shape
