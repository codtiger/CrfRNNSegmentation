from __future__ import print_function
import time
import numpy as np
import sys
import logging
import os
from PIL import Image


class VggUtils:
    def __init__(self, path):
        self.IMAGENET_MEANS = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB mean values

        if os.path.isfile(path):
            self.weights_dict = np.load(path, encoding='latin1').item()
        else:
            logging.error('The path specified does not exist:{}'.format(path))

    def get_weight(self, layer_name):
        if layer_name not in self.weights_dict.keys():
            logging.error('The specified layer does not exist:{}'.format(layer_name))
        else:
            return self.weights_dict[layer_name][0]

    def get_bias(self, layer_name):
        if layer_name not in self.weights_dict.keys():
            logging.error('The specified layer does not exist:{}'.format(layer_name))
        if len(self.weights_dict[layer_name])< 2:
            logging.error('No bias for this layer')
        return self.weights_dict[layer_name][1]

    def reshape_weights(self, shape, layer_name):
        if layer_name not in self.weights_dict.keys():
            logging.error('the specified layer does not exist:{}'.format(layer_name))
        weights = self.weights_dict[layer_name][0]
        # biases = self.weights_dict[layer_name][1]
        if shape is list:
            shape = tuple(shape)
        return np.reshape(weights, shape)
        # TODO: continued...

    def get_average_class(self, num_classes, weights, biases):
        origin_num = weights.shape[3]
        batch = origin_num // num_classes
        new_weights = np.zeros((weights.shape[2], weights.shape[1],
                                weights.shape[0], num_classes), dtype=np.float32)
        new_biases = np.zeros((num_classes,), dtype=np.float32)
        for i in range(0, origin_num, batch):
            next_idx = i + batch
            if next_idx < origin_num:
                new_idx = int(i // batch)
                new_weights[:, :, :, new_idx] = np.mean(weights[:, :, :, i:])
                new_biases[new_idx] = np.mean(biases[i:])
        return new_weights, new_biases

    def get_label_color(self, label):
        return self.PALETTE[label]

    def preprocess_image(self, imgs, save_to_path=None):
        """
        pre-process image by subtracting mean

        :param imgs:image list,must be in bgr format
        :param save_to_path: path to save the output
        :return: pre-processed image
        """
        new_imgs = []
        new_dim = len(imgs) == 1

        for img in imgs:
            assert img.ndim == 3
            img_h, img_w, img_c = img.shape
            assert img_c == 3
            if img_h > 500 and img_w > 500:
                print("Image is larger than 500x500,reduce the size")
            padding_h = 500-img_h
            ph_before, ph_after = padding_h//2, padding_h-padding_h//2
            padding_w = 500-img_w
            pw_before, pw_after = padding_w//2, padding_w-padding_w//2
            new_image = np.pad(img, ((ph_before, ph_after), (pw_before, pw_after), (0, 0)),
                               mode='constant', constant_values=0)
             # TODO: check if it is np.uint8 or float32
            new_image = (new_image - self.IMAGENET_MEANS).astype(np.float32)
            if new_dim:
                new_image = new_image[np.newaxis, :]
            new_imgs.append(new_image)
        if save_to_path is not None:
            np.save(save_to_path, np.array(new_imgs))
        return np.array(new_imgs)


class PascalUtils:

    def __init__(self, path):
        if not os.path.exists(path):
            raise OSError('the path given doesn\'t exist: {} '.format(path))
        dirs = os.listdir(path)
        self.main_path = path
        if not 'JPEGImages' and 'SegmentationClass' in dirs:
            print('this is not a correct path')
        else:
            self.image_path = os.path.join(self.main_path, 'JPEGImages')
            self.label_path = os.path.join(self.main_path, 'SegmentationClass')
            self.label_image_addr = os.listdir(self.label_path)
        self.train_list = []
        self.val_list = []
        self.test_list = []
        self.PALETTE = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128],
                                 [128, 64, 128],
                                 [0, 192, 128],
                                 [128, 192, 128],
                                 [64, 64, 0],
                                 [192, 64, 0],
                                 [64, 192, 0],
                                 [192, 192, 0]])

    def load_split_point(self, path):
        """
        load data points specifying train-valid-test data
        :param path:the path specifying train.txt val.txt files
        :return:
        """
        if not os.path.exists(path):
            raise OSError('the path given doesn\'t exist: {} '.format(path))
        with open(os.path.join(path, 'train.txt'), 'r') as train_file:
            for lines in train_file.readlines():
                self.train_list.append(os.path.join(self.image_path, lines.strip('\n')+'.jpg'))
        with open(os.path.join(path, 'val.txt'), 'r')as val_file:
            for lines in val_file.readlines():
                self.val_list.append(os.path.join(self.image_path, lines.strip('\n')+'.jpg'))

        # self.test_list = [i for i in self.image_path if i not in self.train_list and i not in self.val_list]
        self.test_list = []

    def load_images(self, indices=None, train_valid='train'):

        train_images, val_images, test_images = [], [], []
        if not self.train_list:
            for i in os.listdir(os.path.join(self.image_path, self.label_image_addr)):
                img = np.array(Image.open(i)).astype(np.float32)
                # transform pictures to BGR then add
                train_images.append(img[:, :, ::-1])
            return train_images

        if train_valid == 'train':
            if indices is not None:
                train_indices = indices
            else:
                train_indices = list(np.arange(len(self.train_list)))

            for i in train_indices:
                img = np.array(Image.open(self.train_list[i])).astype(np.float32)
                # transform pictures to BGR then add
                train_images.append(img[:, :, ::-1])
            return train_images

        elif train_valid == 'valid':

            if indices is not None:
                valid_indices = indices
            else:
                valid_indices = list(np.arange(len(self.val_list)))

            for i in valid_indices:
                img = np.array(Image.open(self.val_list[i])).astype(np.float32)
                # transform pictures to BGR then add
                val_images.append(img[:, :, ::-1])
            return val_images
        else:
            for i in self.test_list:
                img = np.array(Image.open(i)).astype(np.float32)
                # transform pictures to BGR then add
                test_images.append(img[:, :, ::-1])
            return test_images

    def load_labels(self, indices=None, train_valid='train'):

        train_labels, val_labels, test_labels = [], [], []
        if not self.train_list:
            for i in os.listdir(os.path.join(self.image_path, self.label_image_addr)):
                img = np.array(Image.open(i)).astype(np.float32)
                # transform pictures to BGR then add
                train_labels.append(img[:, :, ::-1])
            return train_labels

        if train_valid == 'train':
            if indices is not None:
                train_indices = indices
            else:
                train_indices = list(np.arange(len(self.train_list)))
            for i in train_indices:
                img = np.array(Image.open(self.train_list[i])).astype(np.float32)
                # transform pictures to BGR then add
                train_labels.append(img[:, :, ::-1])
            return train_labels

        elif train_valid == 'valid':

            if indices is not None:
                valid_indices = indices
            else:
                valid_indices = list(np.arange(len(self.val_list)))

            for i in valid_indices:
                img = np.array(Image.open(self.val_list[i])).astype(np.float32)
                # transform pictures to BGR then add
                val_labels.append(img[:, :, ::-1])
            return val_labels
        else:
            for i in self.test_list:
                img = np.array(Image.open(i)).astype(np.float32)
                # transform pictures to BGR then add
                test_labels.append(img[:, :, ::-1])
            return test_labels

    def probs_to_label(self, probs, height, width):

        labels = probs.argmax(axis=2).astype(np.uint8)
        # label_image = Image.fromarray(labels, 'P')
        # label_image.putpalette(self.PALETTE)
        # print (np.array(label_image))
        label_image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height * width):
            label_image[i // height, i % height, :] = self.PALETTE[labels[i // height, i % height]]
        # label_image[:, :, :] = self.PALETTE[labels.ravel()].reshape(height, width, 3)

        return label_image

    def get_size(self, train_valid):

        if train_valid == 'train':
            return len(self.train_list)

        else:
            return len(self.val_list)

    def label_to_probs(self, labels, num_classes=21, save_to_path=None):
        """

        :param labels: list of images with probabilities
        :param num_classes: number of classes to choose from palette
        :param save_to_path: path to save the output
        :return:one-hot encoding of images for each pixel

        """
        # size = len(labels)
        one_hot_list = []
        for label in labels:
            pad_x, pad_y = 500-label.shape[0], 500-label.shape[1]
            label = np.pad(label, ((pad_x//2, pad_x-pad_x//2), (pad_y//2, pad_y - pad_y//2), (0, 0)),
                           mode='constant', constant_values=0)
            label = label[:, :, ::-1]
            shape1, shape2 = label.shape[0], label.shape[1]
            new_label = label.reshape(-1, 3)
            # for i in range(size):
            replica = new_label[:, np.newaxis].repeat(num_classes, axis=1)
            one_hot = np.all(replica == self.PALETTE[:num_classes], axis=2).reshape(label.shape[0], label.shape[1], -1)
            one_hot_list.append(one_hot)
        if save_to_path is not None:
            np.save(save_to_path, np.packbits(np.array(one_hot_list)))
        return np.array(one_hot_list)

    def get_label_weights(self):

        weights = np.ones(21, dtype=np.float32)
        weights[0] /= 1000
        return weights / np.sum(weights)
