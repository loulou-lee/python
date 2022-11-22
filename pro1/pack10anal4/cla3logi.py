# pima-indians-diabetes dataset으로 당뇨병 유무 분류 모델
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics._scorer import accuracy_scorer

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/pima-indians-diabetes.data.csv"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pandas.read_csv(url, names=names, header=None)
print(df.head(3), df.shape) # (768, 9)

array = df.values
print(array)
x = array[:, 0:8]
y = array[:, 8]
print(x.shape, y.shape) # (768, 8) (768,)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=7)
print(x_train.shape, x_test.shape) # (537, 8) (231, 8)

model = LogisticRegression()
model.fit(x_train, y_train)
print('예측값 : ', model.predict(x_test[:10]))
print('예측값 : ', y_test[:10])
print((model.predict(x_test) != y_test).sum()) # 58개를 틀림
print('test로 검정한 분류 정확도 : ', model.score(x_test, y_test))
print('train로 검정한 분류 정확도 : ', model.score(x_train, y_train)) # 둘의 차이가 크면 좋지 않다.

from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
print('분류 정확도 : ', accuracy_score(y_test, pred))

print('-----------------')
import joblib
import pickle

# 학습이 끝 난 모델은 저장 후 읽어 사용하도록 함
# joblib.dump('pima_model.sav')
pickle.dump(model, open('pima_model.sav', 'wb'))

# mymodel = joblib.load('pima_model.sav') 
mymodel = pickle.load('pima_model.sav', 'rb')
print('test로 검정한 분류 정확도 : ', mymodel.score(x_test, y_test))

# 새로운 값으로 예측
print(x_test[:1])
print(mymodel.predict([[ 1., 90., 62., 12., 43., 27.2, 0.58, 24. ]]))

