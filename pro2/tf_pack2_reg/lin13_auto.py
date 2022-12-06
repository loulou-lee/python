# AutoMPG dataset으로 자동차 연비 예측 모델
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers

dataset = pd.read_csv('../testdata/auto-mpg.csv')
print(dataset.head(2))
del dataset['car name']

print(dataset.corr())
dataset.drop(['', '', ''], axis='columns')
print(dataset.info())
print(dataset.corr())

# print(dataset.isna().sum()) # horsepower 6
dataset = dataset.dropna()
# print(dataset.isna().sum())

# 시각화
# sns.pairplot(dataset, diag_kind='kde')
# sns.pairplot(dataset[['mpg','cylinders','displacement','horsepower','weight','acceleration']], diag_kind='kde')
# plt.show()

# train / test split
train_dataset = dataset.sample(frac=0.7, random_state=123)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset.shape, test_dataset.shape)

# 표준화
train_stat = train_dataset.describe()
print(train_stat)
train_stat.pop('mpg') # mpg는 label로 사용할 것이므로...
train_stat = train_dataset.transpose()
print(train_stat)

train_label = test_dataset.pop('mpg')
print(train_label[:2])
test_label = test_dataset.pop('mpg')
print(test_label[:2])

def st_func(x): # 표준화 함수
    return (x - train_stat['mean']) / train_stat['std']

# print(st_func(10))
st_train_data = st_func(train_dataset)
st_test_data = st_func(test_dataset)
print(st_train_data.columns)

print()
# model
from keras.models import Sequential
from keras.layers import Dense

def build_model():
    network = Sequential([
            Dense(units=64, activation='relu', input_shape=[7]),
            Dense(units=64, activation='relu'),
            Dense(units=1, activation='linear')
        ])
    
    opti = tf.keras.optimizers.RMSprop(0.001)
    network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
    
    return network

model = build_model()
print(model.summary())

# fit() 전에 모델을 실행해도 됨. 다만 성능은 기대하지 않음
# print(model.predict(st_train_data[:1]))

epochs = 1000
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5) # 5회 이내 더 떨어지지 않으면 조기종료

history = model.fit(st_train_data, train_label, batch_size=32, epochs=epochs,
                    validation_split=0.2, verbose=2, callbacks=[early_stop])

df = pd.DataFrame(history.history)
print(df.head(3))

def plot_history(history):

  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))
  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
  plt.legend()
  plt.show()

plot_history(history)

# 모델 평가
loss, mae, mse = model.evaluate(st_test_data, test_label)
print('loss : {:5.3f}'.format(loss))
print('mae : {:5.3f}'.format(mae))
print('mse : {:5.3f}'.format(mse))

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(test_label, model.predict(st_test_data)))

