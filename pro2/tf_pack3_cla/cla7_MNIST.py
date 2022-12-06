# MNIST dataset (손글씨 이미지 데이터)으로 숫자 이미지 분류 모델 작성
# MNIST는 숫자 0부터 9까지의 이미지로 구성된 손글씨 데이터셋입니다.
# 총 60,000개의 훈련 데이터와 레이블, 총 10,000개의 테스트 데이터와 레이블로 구성되어져 있습니다.
# 레이블은 0부터 9까지 총 10개입니다.
from tf_pack3_cla.cla3_wine import mymodel

'''
★ 자연어처리, 이미지분류는 Dense가 하는 것이지 CNN이 하는거 아니다!!

Dense가 원본을 가지고 하려니 메모리를 많이 잡아먹어서
이미지 원본의 특징만을 뽑아서 특징만 잡아주고 Dense에 밀어주면 Dense가 그걸로 작업을 한다.
'''

import tensorflow as tf
import numpy as np
import sys

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(x_train[0], x_train[0].shape)
print(y_train[0])
# print(x_train[0])
# print(y_train[0])
# for i in x_train[0]:
#     for j in i:
#         sys.stdout.write('%s '%j)
#     sys.stdout.write('\n')

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1])
# plt.show()

# 이미지 분류 지도학습 가능 라벨링돼있다

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
print(x_train[0], x_train[0].shape)

# feature data 를 정규화
x_train /= 255.0
x_test /= 255.0
print(x_train[0])
print(y_train[0], set(y_train))

# label은 원핫 처리 - softmax(다항분류)를 사용하니까
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# train data의 일부를 validation data로 사용하기
x_val = x_train[50000:60000] # 10000개는 validation
y_val = y_train[50000:60000]

x_train = x_train[0:50000] # 50000개는 train
y_train = y_train[0:50000]
print(x_val.shape, x_train.shape)

# model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

model = Sequential()

# model.add(Dense(units=128, input_shape=(784,)))
# model.add(Flatten(input_shape=(28,28))) # reshape을 하지 않은 경우 Flatten에 의해 784로 차원이 변경됨
# model.add(Dense())
# model.add(Activation('relu')) # reshape해서 주석 처리

'''
model.add(Dense(units=128, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=10)) # 출력층
model.add(Activation('softmax'))
'''
model.add(Dense(units=128, input_shape=(784,), Activation('relu')))
model.add(Dropout(rate=0.2))

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2, validation_data=(x_val, y_val))
print(history.history.keys())
print('loss : ', history.history['loss'])
print('val_loss : ', history.history['val_loss'])
print('accuracy : ', history.history['accuracy'])
print('val_accuracy : ', history.history['val_accuracy'])

# 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.show()

# 모델 평가
# score = model.evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict)

model.save('cla7model.hdf5')

# 여기서부터는 저장된 모델로 새로운 데이터에 대한 이미지 분류 작업 진행
mymodel = tf.keras.models.load_model('cla7model.hdf5')

pred = mymodel.predict(x_train[:1])
print('pred : ', pred)
print('예측값 : ', np.argmax(pred, 1))
print('실제값 : ', y_train[:1])
print('실제값 : ', np.argmax(y_train[:1], 1))

