# titanic dataset으로 LogisticRegression, DecisionTree, RandomForest 분류 모델 비교

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../testdata/titanic_data.csv")
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Null 처리 : 평균, 0, 'N' 등으로 변경
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
print(df.head(2))
# print(df.isnull().sum())

# Dtype : object - Sex, Cabin, Embarked 값들의 상태를 분류해서 보기
print('Sex:', df['Sex'].value_counts()) # male 577, female 314
print('Cabin:', df['Cabin'].value_counts()) # Cabin 값들이 너무 복잡하므로 간략하게 정리 - 앞글자만 사용하기로 함
print('Embarked:', df['Embarked'].value_counts())

df['Cabin'] = df['Cabin'].str[:1]
print(df.head(5))

print()
# 성별이 생존확률에 어떤 영향을 주었나?
print(df.groupby(['Sex', 'Survived'])['Survived'].count())
print(233 / (81 + 233)) # 74.2
print(109 / (468 + 109)) # 18.8
# 성별 생존 확률에 대한 시각화
sns.barplot(x='Sex', y='Survived', data=df, ci=95)
plt.show()

# 나이별, Pclass가 생존확률에 어떤 영향을 주었나? ...

print()
# 문자열(object) 데이터를 숫자형으로 변환(범주형)하기
from sklearn import preprocessing

def labelFunc(datas):
    cols = ['Cabin', 'Sex', 'Embarked']
    for c in cols:
        lab = preprocessing.LabelEncoder()
        lab = lab.fit(datas[c])
        datas[c] = lab.transform(datas[c])
    return datas
    
    
df = labelFunc(df)
print(df.head(3))
print(df['Cabin'].unique()) # [7 2 4 6 3 0 1 5 8]
print(df['Sex'].unique()) # [1 0]
print(df['Embarked'].unique()) # [3 0 2 1]

print()
feature_df = df.drop(['Survived'], axis='columns')
print(feature_df.head(2))
label_df = df['Survived']
print(label_df.head(2))

train_x, test_x, train_y, test_y = train_test_split(feature_df, label_df, test_size=0.2, random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

print('---' * 50)
# LogisticRegression, DecisionTree, RandomForest 분류 모델 비교
logmodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(train_x, train_y)
decmodel = DecisionTreeClassifier().fit(train_x, train_y)
rfmodel = RandomForestClassifier().fit(train_x, train_y)

logpredict = logmodel.predict(test_x)
print('LogisticRegression acc : {0:.5f}'.format(accuracy_score(test_y, logpredict)))
decpredict = decmodel.predict(test_x)
print('LogisticRegression acc : {0:.5f}'.format(accuracy_score(test_y, decpredict)))
rfpredict = rfmodel.predict(test_x)
print('LogisticRegression acc : {0:.5f}'.format(accuracy_score(test_y, rfpredict)))
