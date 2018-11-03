import numpy as np
from keras.models import Model
from keras.layers import Softmax, Conv2D, Input
from keras.optimizers import Adam, RMSprop
from CrfRnn import CrfRnn
import logging
import sys
import DataGenerator


class FullyConnected:

    def __init__(self, fcn_model, num_classes):

        self.fcn = fcn_model
        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.model = None

    def prepare_model(self):

        out_fcn = self.fcn.prepare_model()
        image_dims = (500, 500, 3)
        crf_out = CrfRnn(image_dims=image_dims, num_classes=self.num_classes,
                         theta_alpha=0.1, theta_beta=0.1, theta_gamma=1, trainable=True,
                         num_iterations=10)([out_fcn, self.fcn.model.input])

        self.model = Model(input=self.fcn.model.input, output=crf_out)

        return crf_out, self.model

    def predict(self, inputs):

        inputs, prediction = self.fcn.predict(inputs)

        if self.model is None:
            logging.error('model is not prepared yet')
            sys.exit(1)
        return self.model.predict(inputs)

    def get_predict_img(self, prediction):
        return self.fcn.get_predict_img(prediction)

    def train(self, batch_size, epochs, shuffle, learning_rate, momentum, decay, data_split_path=None):
        if data_split_path is not None:
            self.fcn.pascal.load_split_point(data_split_path)

        logging.info("train data generator object getting ready")
        train_generator = DataGenerator(batch_size=batch_size, pascal_object=self.fcn.pascal,
                                        pre_process_func=self.vgg.preprocess_image,
                                        one_hot_func=self.pascal.label_to_probs, num_classes=21,
                                        train_valid='train', shuffle=shuffle)
        logging.info("valid data generator object getting ready")
        val_generator = DataGenerator(batch_size=10, pascal_object=self.fcn.pascal,
                                      pre_process_func=self.vgg.preprocess_image,
                                      one_hot_func=self.pascal.label_to_probs, num_classes=21,
                                      train_valid='valid', shuffle=shuffle)
        if self.model is None:
            logging.error('Model is not initialized')
            sys.exit(1)
        adam = Adam(lr=learning_rate, beta_1=momentum, beta_2=0.99, decay=decay)
        self.model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
        history = self.model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs,
                                           use_multiprocessing=False, workers=3, max_queue_size=1)
        return history
