# 날씨 정보로 나이브베이즈 분류기 작성 - 비 예보
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics._scorer import accuracy_scorer
from pack10anal4.cla10randomforest import cross_vali

df = pd.read_csv("../testdata/weather.csv")
print(df.head(3))
print(df.info())

feature = df[['MinTemp', 'MaxTemp', 'Rainfall']]
label = df['RainTomorrow'].apply(lambda x:1 if x == 'Yes' else 0)
label = df['RainTomorrow'].map({'Yes':1, 'No':0})
print(feature[:3])
print(label[:3])
print(set(label)) # {0, 1}

# 7 : 3 split
train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

gmodel = GaussianNB()
gmodel.fit(train_x, train_y)

pred = gmodel.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', test_y[:10].values)

acc = sum(test_y == pred) / len(pred)
print('acc : ', acc)
print('acc : ', accuracy_score(test_y, pred))

# kfold
from sklearn import model_selection
cross_val = model_selection.cross_val_score(gmodel, feature, label, cv=5)
print('교차 검증 : ', cross_val)
print('교차 검증 평균 : ', cross_val.mean())

print('새로운 자료로 분류 예측')
import numpy as np
new_weather = np.array([[8.0, 24.3, 0.0], [10.0, 25.3, 10.0], []])
print(gmodel.predict(new_weather))