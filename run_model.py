from __future__ import print_function
from FCN8_vgg import *
import cv2

import argparse
import platform
import os

from FullyConnected import FullyConnected

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='Datasets/VOCdevkit/VOC2010',
                   help='Path to the dataset')
args = parser.parse_args()

if not args['data_path']:
    print ("No dataset path specified.")
    sys.exit(1)
# platform = platform.system()

# if platform == 'Linux':
#     pascal_path = '/home/rsn/Datasets/VOC2010'
#     vgg_path = 'vgg16.npy'
# elif platform == 'Darwin':
#     pascal_path = '/Users/apple/Downloads/VOCdevkit/VOC2010'
#     vgg_path = '/Users/apple/Downloads/vgg16.npy'
# elif platform == 'Windows':
#     pascal_path = 'E:\\VOC2010'


def main():
    # fcn-8
    # fcn = FCN8(vgg_path=vgg_path,
    #            pascal_path=pascal_path)
    # fcn.prepare_model()
    #
    # for layer in fcn.model.layers:
    #     if 'input' in layer.name:
    #         continue
    #     elif 'pool' in layer.name and len(layer.name) == 5:
    #         continue
    #     elif 'dropout' in layer.name:
    #         continue
    #     elif 'add' in layer.name:
    #         continue
    #     elif 'softmax' in layer.name:
    #         continue
    #     print(layer.name)
    #     fcn.set_weights(layer.name)

    img = cv2.imread(os.path.join(pascal_path, 'JPEGImages/2007_000129.jpg'))
    # history = fcn.train(5, 1, True, 1e-3, 0.99, 0.9,
    #                     data_split_path=os.path.join(pascal_path, 'ImageSets/Segmentation'))
    # _, prediction = fcn.predict(img)
    # prediction_img = fcn.get_predict_img(prediction)

    # prediction_img = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)

    full_model = FullyConnected(21, vgg_path, pascal_path)
    full_model.prepare_model()
    for layer in full_model.model.layers:
        if 'input' in layer.name:
            continue
        elif 'pool' in layer.name and len(layer.name) == 5:
            continue
        elif 'dropout' in layer.name:
            continue
        elif 'add' in layer.name:
            continue
        elif 'softmax' in layer.name:
            continue
        elif 'crf_rnn' in layer.name:
            continue
        print(layer.name)
        full_model.set_weights(layer.name)
    history = full_model.train(10, 5, True, 1e-6, 0.99, 0.9,
                               data_split_path=os.path.join(pascal_path, 'ImageSets/Segmentation'))
    img, full_prediction = full_model.predict(img)
    full_prediction_img = full_model.get_predict_img(full_prediction)

    full_prediction_img = cv2.cvtColor(full_prediction_img, cv2.COLOR_BGR2RGB)

    cv2.imshow('full-model', full_prediction_img)

    # cv2.imshow('full-model', full_prediction_img)
    cv2.waitKey()
    print (history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['validation', 'train'], loc='upper left')
    plt.show()

    # next plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['validation', 'train'], loc='upper left')
    plt.show()
if __name__ == '__main__':
    main()

