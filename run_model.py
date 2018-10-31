from __future__ import print_function
from FCN8_vgg import *
import cv2
import platform
import os
platform = platform.system()

if platform == 'Linux':
    pascal_path = '/home/rsn/Datasets/VOC2010'
elif platform == 'Darwin':
    pascal_path = '/Users/Downloads/VOCdevkit/VOC2010'
elif platform == 'Windows':
    pascal_path = 'E:\\VOC2010'


def main():

    vgg_path = 'vgg16.npy'
    fcn = FCN8(vgg_path=vgg_path,
               pascal_path=pascal_path)
    fcn.prepare_model()
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

    img = cv2.imread(os.path.join(pascal_path,'JPEGImages/2007_000129.jpg'))
    # img = np.transpose(img, [0, 1, 2])
    weights_before = fcn.get_weights('conv3_3')
    history = fcn.train(5, 1, True, 1e-3, 0.99, 0.9,
                        data_split_path=os.path.join(pascal_path,'ImageSets/Segmentation'))
    weights_after = fcn.get_weights('conv3_3')

    print(np.all(weights_after[0] == weights_before[0]))
    prediction = fcn.predict(img)
    prediction_img = fcn.get_predict_img(prediction)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

    cv2.imshow('me', prediction_img)
    cv2.waitKey()
if __name__ == '__main__':
    main()

