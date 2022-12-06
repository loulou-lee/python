# 문제2)
#
# https://github.com/pykwon/python/tree/master/data
# 자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
# 모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.
# 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력하시오.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection._split import train_test_split
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/data/train.csv", 
                delimiter=',')
df = df.drop(columns='datetime', axis=1)

print(df.corr())

x_data = df[['temp', 'atemp', 'humidity', 'casual', 'registered']]
y_data = df['count']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_data = scaler.fit_transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
print(x_train[:2], x_train.shape)
print(x_test[:2], x_test.shape)

x_train = x_train.astype('float64')

print('-----')
model = Sequential()
model.add(Dense(units=1, input_dim=5, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=200, verbose=0)
print('train/test 후 평가', model.evaluate(x_test, y_test, verbose=0))

pred = model.predict(x_test)
from sklearn.metrics import r2_score
print('train/test 후 결정계수 : ', r2_score(y_test, pred))

# train/test 전 모델로 시각화
plt.plot(y_data, 'b')
plt.plot(pred, 'r--')