# 내가 그린 손글씨 이미지 분류 결과 확인
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 이미지 확대/축소 가능
import tensorflow as tf

im = Image.open('4.png')
img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert('L')) # 그레이스케일
print(img.shape)

# plt.imshow(img, cmap='Greys')
# plt.show()

data = img.reshape([1, 784])
# print(data)
data = data / 255.0 # 정규화
# print(data)

# 학습이 끝난 모델로 내 이미지를 판별
mymodel = tf.keras.models.load_model('cla7model.hdf5')
pred = mymodel.predict(data)
print('pred : ', np.argmax(pred, 1))
