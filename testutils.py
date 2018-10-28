import util
import utils
import numpy as np
from PIL import Image
import cv2
import os
np.random.seed(100)
img = cv2.imread('/Users/apple/Downloads/VOCdevkit/VOC2010/SegmentationClass/2007_001586.png')
# util_result = util.get_label_image(img,img.shape[0], img.shape[1])
vgg = utils.VggUtils('/Users/apple/Downloads/vgg16.npy')
voc = utils.PascalUtils('/Users/apple/Downloads/VOCdevkit/VOC2010')
voc.label_to_prob(img)
# utils_result = vgg.get_label_image(img, img.shape[0], img.shape[1])
# util_result.save('bullshit.png')
# util_result = cv2.imread('bullshit.png')
# utils_result = cv2.cvtColor(utils_result, cv2.COLOR_BGR2RGB)
# print utils_result[300,30]
# cv2.imshow('util', util_result)
# cv2.imshow('utils', utils_result)
# cv2.imshow('difference',util_result-utils_result)
# cv2.waitKey()
# os.remove('bullshit.png')
