# diabetes 데이터로 이항분류(sigmoid)와 다항분류(softmax) 처리

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = np.loadtxt("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/diabetes.csv", delimiter=',')
print(dataset[:1])
print(dataset.shape)

# 이항분류
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 0:8], dataset[:, -1], test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Dense(units=64, input_dim=8, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
scores = model.evaluate(x_test, y_test)
print('%s : %.2f'%(model.metrics_names[0], scores[0]))
print('%s : %.2f'%(model.metrics_names[1], scores[1]))
# 예측값 구하기
# print(y_train[0])
new_data = [[-0.0588235,0.20603,0.,0.,0.,-0.105812,-0.910333,-0.433333]]
pred = model.predict(new_data, batch_size=32, verbose=0)
print('예측결과 : ', pred)
print('예측결과 : ', np.where(pred > 0.5, 1, 0))

print('----------')
# 다항 분류는 label을 원핫인코딩 후 학습에 참여
print(y_train[:3])
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model2 = Sequential()
model2.add(Dense(units=64, input_dim=8, activation='relu'))
model2.add(Dense(units=32, activation='relu'))
model2.add(Dense(units=2, activation='softmax')) # label의 카테고리 수 만큼 결과는 확률 값으로 출력

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # 이항분류 binary_crossentropy 다항분류 categorical_crossentropy
model2.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
scores2 = model2.evaluate(x_test, y_test)
print('%s : %.2f'%(model2.metrics_names[0], scores2[0]))
print('%s : %.2f'%(model2.metrics_names[1], scores2[1]))

new_data2 = [[-0.0588235,0.20603,0.,0.,0.,-0.105812,-0.910333,-0.433333]]
pred2 = model2.predict(new_data2, batch_size=32, verbose=0)
print('예측결과2 : ', pred2)
print('예측결과2 : ', np.argmax(pred2))

