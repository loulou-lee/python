
# 조건부 확률 P(Label|Feature) = P(Feature|Label) * P(Label) / P(Feature)
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics

x = np.array([1,2,3,4,5])
x = x[:, np.newaxis]
print(x)
y = np.array([1,3,5,7,9])

model = GaussianNB().fit(x, y)
print(model)
pred = model.predict(x)
print('분류 정확도 : ', metrics.accuracy_score(y, pred))

# 새로운 값으로 예측
new_x = np.array([[0.1],[0.5],[5],[12]])
new_pred = model.predict(new_x)
print('새로운 예측 결과 : ', new_pred)

# ...
