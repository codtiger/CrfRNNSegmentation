import os
import time
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import high_dim_filter_loader

custom_module = high_dim_filter_loader.custom_module


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

    @staticmethod
    def _comp_mat_initializer(self, input_shape):

        return -1 * np.eye(input_shape, dtype=np.float32)

    def build(self, input_shape):

        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=tf.random_uniform_initializer)
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes,self.num_classes),
                                                     initializer=tf.truncated_normal_initializer)
        self.compatablity_matrix = self.add_weight(name='compatablity_matrix',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=self._comp_mat_initializer)
        super(CrfRnn, self).build(input_shape)

    def call(self, inputs):

        unaries = tf.transpose(inputs[0][0, :, :, :], perm=[2, 1, 0])
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=[2, 1, 0])

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        spatial_norm_values = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                            theta_gamma=self.theta_gamma)
        bilateral_norm_values = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                              theta_alpha=self.theta_alpha,
                                                              theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            normalized = tf.nn.softmax(q_values, axis=0)

            spatial_out = custom_module.high_dim_filter(normalized, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            bilateral_out = custom_module.high_dim_filter(normalized, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            spatial_out = spatial_out / spatial_norm_values
            bilateral_out = bilateral_out / bilateral_norm_values

            message_passing = tf.add(tf.matmul(self.spatial_ker_weights, spatial_out),
                                     tf.matmul(self.bilateral_ker_weights, bilateral_out))

            pairwise = tf.matmul(self.compatablity_matrix, message_passing)

            pairwise = tf.reshape(pairwise, (c, h, w))

            q_values = unaries - pairwise

        # TODO:maybe add a softmax layer after this for future optimization

        return tf.tranpose(tf.reshape(q_values, (1, c, h, w)), (0, 2, 3, 1))

    def compute_output_shape(self, input_shape):

        return input_shape
