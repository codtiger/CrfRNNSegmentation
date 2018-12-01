import numpy as np
from keras.models import Model
from keras.layers import Softmax, Conv2D, Input
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from CrfRnn import CrfRnn
from UpsampleLayer import UpSampleLayer
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add , Lambda , Softmax
import utils
import tensorflow as tf
import logging
import sys
from DataGenerator import DataGenerator


class FullyConnected:

    def __init__(self, num_classes, vgg_path, pascal_path):

        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.model = None
        self.vgg = utils.VggUtils(vgg_path)
        self.learned_fc8_path = '/Users/apple/Downloads/crfasrnn_keras/crfrnn_keras_model.h5'
        self.pascal = utils.PascalUtils(pascal_path)

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

        score_pool4 = Conv2D(21, (1, 1), name='score-pool4', trainable=True)(pool4)
        # score_pool4c = Cropping2D((5, 5))(score_pool4)
        score_fused = Add()([score2, score_pool4])
        # score4 = self.upsample(score_fused, 'score4', 4, 63, 21, 2, False)
        score4 = UpSampleLayer(4, 63, 21, 2, name='score4', trainable=True)(score_fused)
        score_pool3 = Conv2D(21, (1, 1), name='score-pool3', trainable=True)(pool3)
        score_final = Add()([score_pool3, score4])
        # final_upsample = self.upsample(score_final, 'finalupsample', 16, width, 21, 8, False)
        final_upsample = UpSampleLayer(16, width, 21, 8, name='final-score', trainable=True)(score_final)
        crf_out = CrfRnn(image_dims=(height, width, 3), num_classes=self.num_classes,
                         theta_alpha=0.1, theta_beta=0.1, theta_gamma=1, trainable=True,
                         num_iterations=10)([final_upsample, input_img])
        norm_out = Softmax(axis=-1)(crf_out)
        self.model = Model(input=input_img, output=norm_out)

        return crf_out, self.model

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
        return img, self.model.predict(img, verbose=False)

    def get_predict_img(self, prediction):
        prediction = prediction.squeeze()
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
                logging.info('use trained weights')
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
            # weights, biases = self.__get_trained_weights(layer_name)
        self.model.get_layer(layer_name).set_weights([weights, biases])

    def set_score_weight(self, output_shape, input_shape):

        bias_shape = output_shape[3]
        return np.random.randn(output_shape[0], output_shape[1], output_shape[2], output_shape[3]), np.zeros(bias_shape)

    def __get_trained_weights(self, name):
        import h5py
        h5_file = h5py.File(self.learned_fc8_path)
        weights, biases = list(h5_file[name][name].values())[1].value, list(h5_file[name][name].values())[0].value
        return weights, biases

    def get_result_prob(self):
        pass

    def train(self, batch_size, epochs, shuffle, learning_rate, momentum, decay, data_split_path=None):
        if data_split_path is not None:
            self.pascal.load_split_point(data_split_path)

        logging.info("train data generator object getting ready")
        train_generator = DataGenerator(batch_size=batch_size, pascal_object=self.pascal,
                                        pre_process_func=self.vgg.preprocess_image,
                                        one_hot_func=self.pascal.label_to_probs, num_classes=21,
                                        train_valid='train', shuffle=shuffle)
        logging.info("valid data generator object getting ready")
        val_generator = DataGenerator(batch_size=10, pascal_object=self.pascal,
                                      pre_process_func=self.vgg.preprocess_image,
                                      one_hot_func=self.pascal.label_to_probs, num_classes=21,
                                      train_valid='valid', shuffle=shuffle)
        if self.model is None:
            logging.error('Model is not initialized')
            sys.exit(1)
        adam = Adam(lr=learning_rate, beta_1=momentum, beta_2=0.99, decay=decay)
        self.model.compile(optimizer=adam, loss=self.custom_loss(self.pascal.get_label_weights()),
                           metrics=['mean_squared_error', 'accuracy'])
        history = self.model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs,
                                           use_multiprocessing=False, workers=3, max_queue_size=1)
        return history

    def get_weights(self, layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def custom_loss(self, weights):

        weights = K.variable(weights)

        def loss(y_true, y_pred):

            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

            loss_value = - y_true * K.log(y_pred) * weights
            final_loss = K.sum(loss_value, -1)

            return final_loss

        return loss



