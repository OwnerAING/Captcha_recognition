# -*-coding:utf-8 -*-

import numpy as np
from PIL import Image
from hmmlearn import hmm
import os

path = os.path.abspath(os.path.join(os.getcwd(), ".."))


# 提高图像灰度值
def enhance_num(state):
    state[state == 0] = 0
    state[state == 1] = 0
    state[state == 2] = 255
    state[state == 3] = 0
    state[state == 4] = 0
    state[state == 5] = 255
    return state


# 离线学习，直接读取参数，生成状态序列 --list
def get_state(image, components):
    """
    获得序列的状态
    :param image: np.array类型
    :param components:特征数 int类型
    :return: 序列状态序列
    """

    PI = np.load(r'%s/Model/parameter/start_prob.npy' % path)
    trans = np.load(r'%s/Model/parameter/trans_prob.npy' % path)
    means = np.load(r'%s/Model/parameter/means.npy' % path)
    cov = np.load(r'%s/Model/parameter/covars.npy' % path)
    model = hmm.GaussianHMM(n_components=components, covariance_type='full')

    model.startprob_ = PI
    model.transmat_ = trans
    model.means_ = means
    model.covars_ = cov

    state = model.predict(image.reshape(-1, 1))
    return state


# 获取图片像素值,尺寸
def get_pixel(image):
    """
    获取图像高度、宽度和像素值
    :param image:文件名
    :return:宽度、高度，像素值,[::-1]表示值反转
    """
    blcak_image_path = r'%s/Data/BlackPicture/%s.jpg' % (path, image)
    img = Image.open(blcak_image_path)
    size = img.size
    pixel = list(img.getdata())
    pixel_arr = np.array(pixel)
    pixel_arr.reshape(size[::-1])
    return size[::-1], pixel_arr


# 将状态转变为图像显示
def image_show(state_arr, image_name):
    """
    图片显示与储存
    :param state_arr:  图片像素矩阵 ，类型 ：ndarray
    :param image_name: 图片文件名，类型 ： string
    :return: PIL.Image
    """
    img = Image.fromarray(state_arr).convert("L")
    # img.show()
    img.save('%s/Data/HMMFeatureExtraction/%s.jpg' % (path, image_name))
    return img


