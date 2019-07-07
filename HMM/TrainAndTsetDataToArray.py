# -*-coding:utf-8 -*-

import sys
from keras.utils import to_categorical
from PictureProcessing import *
from HMMHandle import *
import os
from PIL import Image
import numpy as np
# import pickle
# from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.externals import joblib

img_rows, img_cols = 26, 18

def get_train_data(letterOrnumberListPath):
    """
    获取相应字母或数字文件夹下的图片，并转换为 ndarray (需循环调用)
    .reshape(-1,1) 行向量转列向量  -1默认为不作为
    (208L, ) 表示行向量
    :param letterOrnumberListPath: 要读取字母或数字文件夹所在位置
    :return: X ,y ，类型： np.ndarray
    """
    flage = True
    XArrays = None
    labels = []
    label = 0  # 从 0 开始编码
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    letterOrnumberList = os.listdir(r'%s/Data/%s' % (path, letterOrnumberListPath))
    # letterOrnumberList = list(map(lambda x: x.split('.')[0], letterOrnumberList))

    print('--------------开始获取数据--------------')
    i = 1

    for letterOrNumber in letterOrnumberList:
        OneNumberFileList = os.listdir(
            r'%s/Data/%s/%s' % (path, letterOrnumberListPath, letterOrNumber))

        for image in OneNumberFileList:
            image = Image.open(
                r'%s/Data/%s/%s/%s' % (path, letterOrnumberListPath, letterOrNumber, image))
            X = np.asarray(image, dtype='float')  # array
            # y = np.array(label]).reshape(-1, 1)
            y = label
            if flage:
                XArrays = X  # 列向量
                labels.append(y)
                flage = False
            else:
                XArrays = np.row_stack((XArrays, X))
                # labels = np.row_stack((labels, y))
                labels.append(label)
            i += 1
            print('第   %d条   已获取' % i)

        label += 1

    # XArrays /= 255

    labels = np.array(labels)
    print(XArrays.shape)
    print(labels.shape)
    XArrays = XArrays.reshape(labels.shape[0], img_rows, img_cols, 1)
    print(XArrays.shape)
    print('数据获取完毕，开始保存')
    print(XArrays)

    y = to_categorical(labels, num_classes=36)
    XArrays, y = shuffle(XArrays, y)

    joblib.dump(XArrays, r'%s/Data/TrainArray/X.pkl' % path)
    joblib.dump(y, r'%s/Data/TrainArray/y.pkl' % path)

    return XArrays, y


def get_test_data(image, testDataPath='Data/TestData'):

    black_image = image2black(testDataPath, image)

    size, pixel = get_pixel(black_image)
    state = get_state(pixel, 6)
    state_enhance = enhance_num(state)
    state_arr = state_enhance.reshape(size)
    img = image_show(state_arr, image)

    FourNumInOneImage = image_split(testDataPath, img)

    return FourNumInOneImage



