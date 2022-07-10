from __future__ import print_function
import numpy as np
from tensorflow import keras
import gc


class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, pascal_object, pre_process_func, one_hot_func, num_classes, train_valid='train', shuffle=True):
        """

        :param batch_size:
        :param pascal_object:
        :param pre_process_func:
        :param one_hot_func:
        :param train_valid:
        :param shuffle:
        :param num_classes:
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pascal = pascal_object
        self.pre_process = pre_process_func
        self.one_hot = one_hot_func
        self.name = train_valid
        self.num_classes = num_classes
        self.size = self.pascal.get_size(self.name)
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(self.size / self.batch_size))

    def __getitem__(self, index):

        indexes = list(self.indexes[index*self.batch_size:(index+1)*self.batch_size])

        x, y = self.generate_batch(indexes)
        gc.collect()
        return x, y

    def on_epoch_end(self):

        self.indexes = np.arange(self.size)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def generate_batch(self, indices):

        images = self.pascal.load_images(indices=indices, train_valid=self.name)
        labels = self.pascal.load_labels(indices=indices, train_valid=self.name)
        images, labels = self.pre_process(images), self.one_hot(labels, self.num_classes)


        return images, labels
