# 선형회귀모델을 다항회귀모델(다중회귀랑 다름)로 변환
# 선형 가정이 신뢰도가 떨어질 경우 대처방법으로 다항식을 추가할 수 있다.

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

print(np.corrcoef(x, y))

plt.scatter(x, y)
plt.show()

# 선형회귀모델 작성
from sklearn.linear_model import LinearRegression

x = x[:, np.newaxis] # 차원 확대
print(x)
model = LinearRegression().fit(x, y)
ypred = model.predict(x)
print(ypred)
# 결과가 안 좋으므로 다항회귀 분석을한다.
# plt.scatter(x, y)
# plt.plot(x, ypred, c='red')
# plt.show()

# 좀 더 복잡한 형태의 모델을 필요 : 다항식 특징(feature)을 추가한 다항회귀모델 작성 
#열을 추가
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False) # degree=열의 갯수
x2 = poly.fit_transform(x) # 특징행렬 작성
print(x2)

model2 = LinearRegression().fit(x2, y) # 특징행렬 값으로 학습
ypred2 = model2.predict(x2)
print(ypred2)

plt.scatter(x, y)
plt.plot(x, ypred2, c='red')
plt.show()