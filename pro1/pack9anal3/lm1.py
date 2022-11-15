# 회귀분석(선형회귀분석)
# 각각의 데이터에 대한 잔차(예측값 - 실제값)제곱합이 최소가 되는 추세선(표준회귀선)을 만들고, 이를 통해
# 독립변수가 종속변수(y, label)에 얼마나 영향을 주는지 인과관계를 분석
# 독립변수:연속형, 종속변수:연속형. 두 변수는 상관관계가 있어야 하고 나아가서는 인과관계가 있어야 한다.
# 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
from yapf.yapflib.pytree_utils import CopyYapfAnnotations
import pandas as pd

np.random.seed(12)

# 모델 맛보기

# 방법1 : make_regression을 사용. model x
x, y, coef = make_regression(n_samples = 50, n_features=1, bias=100, coef = True)
print(x)
print(y)
print(coef)

# 회귀식 y = a + bx, y = b + wx, y = wx + b
y_pred = 89.47430739278907 * 0.75314283 + 100
print('y_pred : ', y_pred)

y_pred_new = 89.47430739278907 * 33 + 100
print('y_pred_new : ', y_pred_new)

xx = x
yy = y

print()
# 방법2 : LinearRegression을 사용. model 만들어짐
from sklearn.linear_model import LinearRegression

model = LinearRegression()
fit_model = model.fit(xx, yy) # 이미 수집된 학습 데이터로 모형 추정 : 절편, 기울기 얻음(내부적으로 최소 제곱법)
print('기울기(slope, w) : ', fit_model.coef_) # coef 회귀계수, 기울기
print('절편(bias, b) : ', fit_model.intercept_) # y 절편
# 예측값 확인 함수로 미지의 feature에 label을 예측
print(xx[[0]]) # [-1.70073563]
y_new = fit_model.predict(xx[[0]]) # sklearn은 입력값을 2차원으로 입력해야한다.
print('y_new(예측값) : ', y_new)
print('실제값 : ', yy[0])

y_new2 = fit_model.predict([[33]])
print('y_new2(예측값) : ', y_new2)

print()
# 방법3 : ols를 사용. model O - 잔차제곱합(RSS)을 최소화하는 가중치 벡터를 행렬미분으로 구하는 방법 == 최소제곱법
import statsmodels.formula.api as smf
import pandas as pd

print(xx.shape) # (50, 1)
x1 = xx.flatten() # 차원축소함수
print(x1.shape) # (50,)
y1 = yy

data = np.array([x1, y1])
df = pd.DataFrame(data.T)
df.columns = ['x1', 'y1']
print(df.head(3))

model2 = smf.ols(formula='y1 ~ x1', data=df).fit()
print(model2.summary())

# 예측값 확인 함수
print(x1[:2])
new_df = pd.DataFrame({'x1':[-1.70073563 -0.57794537]}) # 기존 자료를 사용하였음
new_pred = model2.predict(new_df)
print('new_pred : \n', new_pred)
print('실제값 : \n', df.y1[:2])

# 전혀 새로운 x 값에 대한 예측
new_df2 = pd.DataFrame({'x1':[33.0, -1.234]})
new_pred2 = model2.predict(new_df2)
print('new_pred2 : \n', new_pred2)

