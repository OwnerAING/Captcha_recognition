# -*-coding:utf-8 -*-
# from PictureProcessing import *
# from HMMHandle import *
from TrainAndTsetDataToArray import *
import os


# rgb_image_list = map(lambda x: x.split('.')[0], os.listdir(r'../Data/captcha'))
black_image_list = map(lambda x: x.split('.')[0], os.listdir(r'../Data/BlackPicture'))
# pad_image_list = map(lambda x: x.split('.')[0], os.listdir(r'../Data/Fields'))
# HMM_image_list = map(lambda x: x.split('.')[0], os.listdir(r'../Data/HMMFeatureExtraction'))

# print '-------image2black开始执行--------'
# i = 0
# # 将图像转换为灰度图像
# for image in rgb_image_list:
#     print '-----开始执行image2black%d张----' % i
#     black_image = image2black('Data', image)
#     i += 1

i = 0
# 利用GM—HMM模型对图像进行标注，识别出所含验证码
for image in black_image_list:
    print('-----执行HMM-trans   %d 张----' % i)
    size, pixel = get_pixel(image)
    state = get_state(pixel, 6)
    state_enhance = enhance_num(state)
    state_arr = state_enhance.reshape(size)
    image_show(state_arr, image)
    i += 1

# i = 0
# # 将标注的图像进行分割
# for image in HMM_image_list:
#     print '-----开始执行image_split  %d张----' % i
#     image_split(image, 4)
#     i += 1


# i = 0
# # 对分割的图像进行扩充 Pad
# for image in pad_image_list:
#     try:
#         print('-----开始执行pad_image   %d张----' % i)
#         pad_image(image)
#         i += 1
#     except:
#         pass


X, y = get_train_data('test')


