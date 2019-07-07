# -*- coding:utf-8 -*-
import numpy as np
from hmmlearn import hmm
import PIL.Image as image
import os
from sklearn.externals import joblib

n = 6
path = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_list = list(map(lambda x: x.split('.')[0], os.listdir(r'../Data/test')))

train_image = image_list[0]
img = image.open(r'%s/Data/BlackPicture/%s.jpg' % (path, train_image))

pixel = np.array(list(img.getdata()))

model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=1000)
model = model.fit(pixel.reshape(-1, 1))
np.set_printoptions(suppress=True)

trans = model.transmat_
start = model.startprob_
mean = model.means_
cov = model.covars_
print('转换概率\n', trans)
print('初始概率\n', start)
print('均值\n', mean)
print('方差\n', cov)

# 保存参数
np.save(r'%s/Model/parameter/start_prob' % path, np.array(start))
np.save(r'%s/Model/parameter/trans_prob' % path, np.array(trans))
np.save(r'%s/Model/parameter/means' % path, np.array(mean))
np.save(r'%s/Model/parameter/covars' % path, np.array(cov))

joblib.dump(model, r'%s/Model/model/HMMmodel.pkl' % path)

