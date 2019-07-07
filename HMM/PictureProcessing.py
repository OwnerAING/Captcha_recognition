# -*-coding:utf-8 -*-

from PIL import Image
import numpy as np
import os


def image2black(dataPath, image):

    """
    :return: image: 要转换的图像
    :return: PIL.Image : 返回灰度图像
    """
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 当前路径的上一级
    img = Image.open(r'%s/%s/captcha/%s.jpg' % (path, dataPath, image))
    black_image = img.convert('L')
    # low_size.show()
    black_image.save(r'%s/%s/BlackPicture/%s.jpg' % (path, dataPath, image))
    return black_image


def pad_image(dataPath, image, new_size=[26, 18]):
    """
    保持原图像在中间位置，四周填充为白色
    :param image: numpy.array, shape = [H,W],分别表示：高，宽
    :param new_size: list (like [H, W])
    :return: numpy.array, shape = new_size
    """
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))

    try:
        fig_image = Image.open(r'%s/%s/Fields/%s.jpg' % (path, dataPath, image))
        open_image = np.asarray(fig_image)
    except:
        open_image = np.asarray(image)

    assert (open_image.shape[0] <= new_size[0]) &\
           (open_image.shape[1] <= new_size[1])
    h_need_pad = int(new_size[0] - open_image.shape[0])
    h_top = int(h_need_pad/2)
    h_bottom = h_need_pad - h_top
    w_need_pad = int(new_size[1] - open_image.shape[1])
    w_left = int(w_need_pad/2)
    w_right = w_need_pad - w_left

    pd_image = np.pad(open_image[:, :],
                               ((h_top, h_bottom), (w_left, w_right)),
                               mode='constant',
                               constant_values=(0, 0))

    Image.fromarray(pd_image).save(r'%s/%s/PadedImage/%s.jpg' % (path, dataPath, image))

    # return pd_image


def image_split(dataPath, image, image_character_num):

    """
    验证码图像分割
    :param image: 要切割的图像
    :param image_character_num: 图像包含的数字+字母个数
    :param image_name : 图像名字， 切分后储存
    :return: none
    """

    IDcodes = None
    FourMumInOneImage = np.array([])
    flag = True
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    try:
        opened_image = Image.open(
            r'%s/%s/HMMFeatureExtraction/%s.jpg' % (path, dataPath, image))  # 打开图像
    except:
        image_array = image
        opened_image = Image.fromarray(image_array)

    start_letter = False  # 找出每个字母开始位置
    end_letter = False  # 找出每个字母结束位置
    start = 0
    # end = 0

    letters = []  # 存储坐标
    for x in range(opened_image.size[0]):
        for y in range(opened_image.size[1]):
            pix = opened_image.getpixel((x, y))
            if pix == 255:
                start_letter = True
        if end_letter == False and start_letter == True:
            end_letter = True
            start = x
        if end_letter == True and start_letter == False:
            end_letter = False
            end = x
            # 切分太宽，说明两个字母切分到一张图像，则认为切分为两份
            if (end - start) > 20:
                letters.append((start, start + (end - start)/2))
                letters.append((start + (end - start)/2, end))
            else:
                letters.append((start, end))

        start_letter = False

    # 因为切割出来的图像有可能是噪声点
    # 筛选可能切割出来的噪声点,只保留开始结束位置差值最大的位置信息
    subtract_array = []  # 存储 结束-开始 值

    for each in letters:
        subtract_array.append(each[1] - each[0])
    reset = sorted(subtract_array, key=lambda x: np.abs(x), reverse=True)[0: image_character_num]

    letter_chioce = []  # 存储 最终选择的点坐标
    for each in letters:
        if int(each[1] - each[0]) in reset:
            letter_chioce.append(each)

    for num, letter in enumerate(letter_chioce):
        im_split = opened_image.crop(
            (letter[0], 0, letter[1], opened_image.size[1]))  # (切割的起始横坐标，起始纵坐标，切割的宽度，切割的高度)
        im_split.save(r'%s/%s/Fields/%s_%d.jpg' % (path, dataPath, image, num))

        im_split_array = np.asarray(im_split).flatten()
        if flag:
            FourMumInOneImage = im_split_array
            flag = False
        else:
            FourMumInOneImage = np.row_stack((FourMumInOneImage, im_split_array))

    return FourMumInOneImage


