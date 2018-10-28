
from __future__ import print_function
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add , Lambda , Softmax
import numpy as np
from keras.optimizers import Adam
import utils
import logging
import os
import sys
from UpsampleLayer import *


class FCN8:

    def __init__(self, vgg_path, pascal_path, wd=0.5, dropout=0.5):
        self.weight_decay = wd
        self.drop = dropout
        self.vgg = utils.VggUtils(vgg_path)
        self.pascal = utils.PascalUtils(pascal_path)
        self.model = None
        self.learned_fc8_path = '/Users/apple/Downloads/crfasrnn_keras/crfrnn_keras_model.h5'
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def prepare_model(self):
        height, width = 500, 500
        input_img = Input((height, width, 3))
        x = input_img

        # x = ZeroPadding2D(padding=(100, 100))(input_img)

        # VGG-16 convolution block 1

        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1', trainable=False)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1', trainable=False)(x)

        # VGG-16 convolution block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', trainable=False)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same', trainable=False)(x)

        # VGG-16 convolution block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', trainable=False)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', trainable=False)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same', trainable=False)(x)
        pool3 = x

        # VGG-16 convolution block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', trainable=False)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', trainable=False)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same', trainable=False)(x)
        pool4 = x

        # VGG-16 convolution block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', trainable=False)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', trainable=False)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same', trainable=False)(x)

        # Fully Connected Layer as Convolutions
        x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc6', trainable=False)(x)
        x = Dropout(0.5)(x)
        x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc7', trainable=False)(x)
        x = Dropout(0.5)(x)
        x = Conv2D(21, (1, 1), padding='same', name='score-fr', trainable=False)(x)
        final = x
        # score2 = self.upsample(x, 'score2', 4, 32, 21, 2, False)
        score2 = UpSampleLayer(4, 32, 21, 2, name='score2', trainable=True)(x)

        score_pool4 = Conv2D(21, (1, 1), name='score-pool4', trainable=False)(pool4)
        # score_pool4c = Cropping2D((5, 5))(score_pool4)
        score_fused = Add()([score2, score_pool4])
        # score4 = self.upsample(score_fused, 'score4', 4, 63, 21, 2, False)
        score4 = UpSampleLayer(4, 63, 21, 2, name='score4', trainable=True)(score_fused)
        score_pool3 = Conv2D(21, (1, 1), name='score-pool3', trainable=True)(pool3)
        score_final = Add()([score_pool3, score4])
        # final_upsample = self.upsample(score_final, 'finalupsample', 16, width, 21, 8, False)
        final_upsample = UpSampleLayer(16, width, 21, 8, name='final-score', trainable=True)(score_final)
        final_soft = Softmax(axis=-1)(final_upsample)
        # final_upsample = Softmax(axis=-1)(final_upsample)
        self.model = Model(input_img, final_soft, name='FCN8_vgg')

        return final_upsample
        # Fully Connected Layer as

    def get_deconv_filter(self, size, out_size):

        factor = size + 1 // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        x_filter, y_filter = np.ogrid[:size, :size]
        weights = (1 - np.abs(x_filter - center) / factor) *\
                  (1 - np.abs(y_filter - center) / factor)
        bilinear_filter = np.ones((size, size, out_size, out_size))
        for i in range(out_size):
            bilinear_filter[:, :, i, i] = weights
        return bilinear_filter.astype(np.float32)

    def upsample(self, input, name, kernel_size, out_shape, class_num, stride=2, train=False):

        bilinear_filter = self.get_deconv_filter(kernel_size, class_num)

        if train:
            transpose_kernel = tf.get_variable(name
                                               , initializer=bilinear_filter, trainable=True)
        else:
            transpose_kernel = tf.get_variable(name
                                               , initializer=bilinear_filter, trainable=False)

        output = tf.nn.conv2d_transpose(input, transpose_kernel, output_shape=[1, out_shape, out_shape, 21]
                                        , strides=[1, stride, stride, 1], padding='SAME')
        return output

    def predict(self, img):
        img = self.vgg.preprocess_image([img])[0]
        if self.model is None:
            logging.error('Model is uninitialized')
            sys.exit(1)
        return self.model.predict(img, verbose=False)

    def get_predict_img(self, img):
        prediction = self.predict(img).squeeze()
        # return self.vgg.get_label_image(prediction, prediction.shape[0], prediction.shape[1])
        return self.pascal.probs_to_label(prediction, prediction.shape[0], prediction.shape[1])

    def set_weights(self, layer_name):
        if self.model is None:
            logging.error('Model is not initialized')
            sys.exit(1)
        if layer_name.find('fc') == 0:
            # shape = self.model.get_layer(layer_name).weights[0].get_shape().as_list()
            # weights = self.vgg.reshape_weights(shape, layer_name)
            # biases = self.vgg.get_bias(layer_name)
            weights, biases = self.__get_trained_weights(layer_name)
        elif 'score-fr' in layer_name:
            # shape = self.model.get_layer(layer_name).weights[0].get_shape().as_list()
            # weights = self.vgg.reshape_weights((4096, 1, 1, 1000), 'fc8')
            # biases = self.vgg.get_bias('fc8')
            # weights, biases = self.vgg.get_average_class(21, weights, biases)
            weights, biases = self.__get_trained_weights(layer_name)
        elif 'score-pool' in layer_name:
            if self.model.get_layer(layer_name).trainable:
                logging.info ('use trained weights')
                weights, biases = self.__get_trained_weights(layer_name)
            else:
                # shape = self.model.get_layer(layer_name).weights[0].get_shape().as_list()
                # input_shape = self.model.get_layer(layer_name).input_shape
                # weights, biases = self.set_score_weight(shape, input_shape)
                weights, biases = self.__get_trained_weights(layer_name)
        elif 'score' in layer_name:
            return

        else:
            weights = self.vgg.get_weight(layer_name)
            biases = self.vgg.get_bias(layer_name)
        self.model.get_layer(layer_name).set_weights([weights, biases])

    def set_score_weight(self, output_shape, input_shape):
        bias_shape = output_shape[3]
        return np.random.randn(output_shape[0], output_shape[1], output_shape[2], output_shape[3]), np.zeros(bias_shape)

    def __get_trained_weights(self, name):
        import h5py
        h5_file = h5py.File(self.learned_fc8_path)
        weights, biases = h5_file[name][name].values()[1].value, h5_file[name][name].values()[0].value
        return weights, biases

    def get_result_prob(self):
        pass

    def train(self, batch_size, epochs, shuffle, learning_rate, momentum, decay, data_split_path=None):
        if os.path.exists('train_data.npy'):
            train_data = np.load('train_data.npy')

        if os.path.exists('valid_data.npy'):
            valid_data = np.load('valid_data.npy')
        else:
            if data_split_path is not None:
                self.pascal.load_split_point(data_split_path)

            train_data, valid_data, test_data = self.pascal.load_images()
            train_data = self.vgg.preprocess_image(train_data, 'train_data.npy')
            valid_data = np.load('valid_data.npy')
            valid_data = self.vgg.preprocess_image(valid_data, 'valid_data.npy')
            # test_data = self.vgg.preprocess_image(test_data, 'test_data.npy')
        logging.info("input data ready,now going for labels")
        if os.path.exists('train_label.npy'):
            self.logger.info('train labels loaded')
            train_label = np.unpackbits(np.load('train_label.npy'))
            train_label = train_label[:(964*500*500*21)].reshape(964, 500, 500, 21)
        if os.path.exists('valid_label.npy'):
            self.logger.info('valid labels loaded')
            valid_label = np.unpackbits(np.load('valid_label.npy'))
            valid_label = valid_label[:(964*500 * 500 * 21)].reshape(964, 500, 500, 21)
        else:

            self.pascal.load_split_point(data_split_path)
            train_label, valid_label, test_label = self.pascal.load_labels()
            logging.info('one hot encoding the labels ')
            train_label = self.pascal.label_to_probs(train_label, 21, 'train_label.npy')
            valid_label = self.pascal.label_to_probs(valid_label, 21, 'valid_label.npy')
        if self.model is None:
            logging.error('Model is not initialized')
            sys.exit(1)
        adam = Adam(lr=learning_rate, beta_1=momentum, beta_2=0.99, decay=decay)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x=train_data, y=train_label, batch_size=batch_size,
                                 epochs=epochs, verbose=1, shuffle=shuffle)
        return history

