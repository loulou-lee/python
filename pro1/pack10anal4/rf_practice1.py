from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing.tests.test_data import n_features
from sklearn.metrics import accuracy_score

df = pd.read_csv("../testdata/winequality-red.csv")
print(df.head(3))
print(df.columns)
print(df.info())
print(df.isnull().any())

df = df.dropna(subset=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']) # feature
print(df.shape) # (714, 12)

# df_x = [['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']] # feature

df_x = df.drop(['quality'], axis='columns')

df_y = df['quality']

# train/test split
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=0)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) #(535, 6) (179, 6) (535,) (179,)
#
#model
model = RandomForestClassifier(n_estimators=500, criterion='entropy')
model.fit(train_x, train_y)
#
pred = model.predict(test_x)
print('예측값 : ', pred[:5])
print('실제값 : ', np.array(test_y[:5]))

print('acc : {0:.5f}'.format(accuracy_score(test_y, pred))
