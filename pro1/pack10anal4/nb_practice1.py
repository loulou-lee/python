import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics._scorer import accuracy_scorer

df = pd.read_csv("../testdata/mushrooms.csv")
print(df.head(3))
print(df.info())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])
print(df.head(3))
print(df.isnull().sum()) # 널값없음

feature = df.drop(['class'], axis=1) # df.iloc[:, 1:23] # 

# https://www.kaggle.com/questions-and-answers/132668

# from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# o_encoder = OrdinalEncoder()
# data_encoded = o_encoder.fit_transform(df[feature])
# df_encoded = pd.DataFrame(data_encoded, columns=feature)
# data_encoded



label = df['class'].apply(lambda x:1 if x == 'p' else 0)
label = df['class'].map({'p':1, 'e':0})
print(feature[:3])
print(label[:3])
# print(set(label)) # {0, 1}
#
# 7 : 3 split
train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
#
gmodel = GaussianNB()
gmodel.fit(train_x, train_y)
#
pred = gmodel.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', test_y[:10].values)
#
# acc = sum(test_y == pred) / len(pred)
# print('acc : ', acc)
# print('acc : ', accuracy_score(test_y, pred))
#
# # kfold
# from sklearn import model_selection
# cross_val = model_selection.cross_val_score(gmodel, feature, label, cv=5)
# print('교차 검증 : ', cross_val)
# print('교차 검증 평균 : ', cross_val.mean())
#
# print('새로운 자료로 분류 예측')
# import numpy as np
# new_weather = np.array([[8.0, 24.3, 0.0], [10.0, 25.3, 10.0], []])
# print(gmodel.predict(new_weather))