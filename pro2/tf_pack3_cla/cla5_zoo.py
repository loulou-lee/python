# zoo animal dataset으로 동물의 type을 종류
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical

xy = np.loadtxt("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/zoo.csv", delimiter=',')
print(xy[0], xy.shape)
x_data = xy[:, 0:-1]
y_data = xy[:, -1]
print(x_data[0])
print(y_data[0], ' ', set(y_data))

# train / test 생략

# label은 one-hot 처리를 해야 함
y_data = to_categorical(y_data) # 명시적인 one-hot 처리
print(y_data[0])

model = Sequential()
model.add(Dense(32, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
print(model.summary())

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# loss='sparse_categorical_crossentropy' 하면 내부적으로 one-hot 처리 해야함 따라서 17, 18행 생략 가능 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=100, batch_size=10, validation_split=0.3, verbose=0)
print('evaluate : ', model.evaluate(x_data, y_data, batch_size=10, verbose=0))

# loss, acc를 시각화
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

import matplotlib.pyplot as plt

plt.plot(loss, 'b-', label='loss')
plt.plot(val_loss, 'r--', label='val_acc')
plt.xlabel('epochs')
plt.legend()
plt.show()

print()
# 예측
pred_data = x_data[:1]
print(model.predict(pred_data))
print('예측값 : ', np.argmax(model.predict(pred_data)))

print()
# 여러 개 예측값
pred_datas = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값 : ', preds)
print('실제값 : ', y_data[:5].flatten())