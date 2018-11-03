from __future__ import print_function
from FCN8_vgg import *
import cv2
import platform
import os
from FullyConnected import FullyConnected
platform = platform.system()

if platform == 'Linux':
    pascal_path = '/home/rsn/Datasets/VOC2010'
    vgg_path = 'vgg16.npy'
elif platform == 'Darwin':
    pascal_path = '/Users/apple/Downloads/VOCdevkit/VOC2010'
    vgg_path = '/Users/apple/Downloads/vgg16.npy'
elif platform == 'Windows':
    pascal_path = 'E:\\VOC2010'


def main():

    fcn = FCN8(vgg_path=vgg_path,
               pascal_path=pascal_path)
    fcn.prepare_model()
    # full_model = FullyConnected(fcn, 21)
    # full_model.prepare_model()
    for layer in fcn.model.layers:
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
        print (layer.name)
        fcn.set_weights(layer.name)

    img = cv2.imread(os.path.join(pascal_path, 'JPEGImages/2007_000129.jpg'))
    weights_before = fcn.get_weights('conv3_3')
    history = fcn.train(5, 1, True, 1e-3, 0.99, 0.9,
                        data_split_path=os.path.join(pascal_path, 'ImageSets/Segmentation'))
    weights_after = fcn.get_weights('conv3_3')

    print(np.all(weights_after[0] == weights_before[0]))
    _, prediction = fcn.predict(img)
    prediction_img = fcn.get_predict_img(prediction)

    prediction_img = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)

    # full_prediction = full_model.predict(img)
    # full_prediction_img = full_model.get_predict_img(full_prediction)

    # full_prediction_img = cv2.cvtColor(full_prediction_img, cv2.COLOR_BGR2RGB)

    cv2.imshow('fcn-8', prediction_img)

    # cv2.imshow('full-model', full_prediction_img)
    cv2.waitKey()
if __name__ == '__main__':
    main()

