from FCN8_vgg import *
import cv2


def main():
    vgg_path = '/Users/apple/Downloads/vgg16.npy'
    pascal_path = '/Users/apple/Downloads/VOCdevkit/VOC2010'
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
        print layer.name
        fcn.set_weights(layer.name)

    img = cv2.imread('/Users/apple/Downloads/VOCdevkit/VOC2010/JPEGImages/2009_005217.jpg')
    # img = np.transpose(img, [0, 1, 2])

    history = fcn.train(10, 3, True, 1e-4, 0.9, 0.98,
                        data_split_path='/Users/apple/Downloads/VOCdevkit/VOC2010/ImageSets/Segmentation')

    prediction = fcn.get_predict_img(img)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

    cv2.imshow('me', prediction)
    cv2.waitKey()
if __name__ == '__main__':
    main()

