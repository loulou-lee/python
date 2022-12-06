# 현대차 가격예측 모댈
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import tensorflow as tf
from dask.array.ufunc import remainder

train_df = pd.read_excel("../testdata/hd_carprice.xlsx", sheet_name="train")
test_df = pd.read_excel("../testdata/hd_carprice.xlsx", sheet_name="test")
print(train_df.head(2))
print(test_df.head(2))

# 전처리 
x_train = train_df.drop(['가격'], axis=1) # feature
x_test = test_df.drop(['가격'], axis=1)
y_train = train_df[['가격']]
y_test = test_df[['가격']]
print()
# make_column_transformer : 여러 개의 열에 대해 OneHotEncoder 처리 가능
transform = make_column_transformer((OneHotEncoder(), ['종류','연료','변속기']), remainder='passthrough')
transform.fit(x_train)
x_train = transform.transform(x_train) # 세 개의 열이 참여해 원핫 수행 후 모든 칼럼을 표준화
print(x_train[:2])
print(x_train.shape)
print(y_train.shape)
x_test = transform.transform(x_test)

# function api 사용
input = tf.keras.layers.Input(shape=(16,))
net = tf.keras.layers.Dense(units=32, activation='relu')(input)
net = tf.keras.layers.Dense(units=32, activation='relu')(net)
net = tf.keras.layers.Dense(units=1, activation='linear')(net)
model = tf.keras.models.Model(input, net)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=2)
print('evaluate : ', model.evaluate(x_test, y_test))
      
y_predict = model.predict(x_test)
print('예측값 : ', y_predict[:5].flatten())
print('실제값 : ', y_test[:5].values.flatten())

print('----GradientTape 객체 사용----')
input = tf.keras.layers.Input(shape=(16,))
net = tf.keras.layers.Dense(units=32, activation='relu')(input)
net = tf.keras.layers.Dense(units=32, activation='relu')(net)
net = tf.keras.layers.Dense(units=1, activation='linear')(net)
model2 = tf.keras.models.Model(input, net)

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

EPOCHS = 50
for epoch_ind in range(EPOCHS):
    with tf.GradientTape() as tape:
        predict = model2(x_train, training=True)
        loss_val = loss(y_train, predict)
        
    gradients = tape.gradient(loss_val, model2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
    
    train_loss.update_state(loss_val)
    predict = model2(x_test)
    loss_val = loss(y_test, predict)
    test_loss.update_state(loss_val)
    print('epoch:{}/{}, train loss:{:.3f}, test loss:{:.3f}'.format(epoch_ind + 1, EPOCHS,
                                                                    train_loss.result().numpy(),
                                                                    test_loss.result().numpy()))
    train_loss.reset_states()
    test_loss.reset_states()
    
y_predict = model2.predict(x_test)
print('예측값 : ', y_predict[:5].flatten())
print('실제값 : ', y_test[:5].values.flatten())

print()
# 새 값으로 자동차 가격 예측
new_data = [[2015, '중형', 5.3, 200, 27.0, '디젤', 0,1500, 1200, '자동']]
new_data = pd.DataFrame(new_data,
                        columns=['년식', '종류', '연비', '마력'])

new_data = transform.transform(new_data)
new_pred = model2.predict(new_data)
print('예측 자동차 가격 : ', new_pred.flatten())
