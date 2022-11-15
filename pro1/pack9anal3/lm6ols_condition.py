# 선형회귀 분석 모델
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

# 여러 매체의 광고비에 따른 판매량(액?) 데이터 사용
advdf = pd.read_csv("../testdata/Advertising.csv", usecols=[1,2,3,4])
print(advdf.head(3), advdf.shape) # (200, 4)
print(advdf.info())

# 단순선형회귀 : tv, sales
print('상관계수(r) : ', advdf.loc[:, ['tv', 'sales']].corr()) # 0.782
# 'tv'가 'sales'에 영향을 준다 라고 가정

# 모델 생성
# lm = smf.ols(formula='sales ~ tv', data = advdf)
# lm = lm.fit()
lm = smf.ols(formula='sales ~ tv', data = advdf).fit()
print(lm.summary())

# 시각화
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('tv')
plt.ylabel('sales')
y_pred = lm.predict(advdf.tv)
# print('y_pred : ', y_pred.values)
# print('real y : ', advdf.sales.values)
plt.plot(advdf.tv, y_pred, c='r') #plt.scatter
plt.show()

# 예측1 : 새로운 tv 값으로 sales를 추정
x_new = pd.DataFrame({'tv':[230.1, 44.5, 100]})
pred = lm.predict(x_new)
print('예측 결과 : ', pred.values)

print('----'*20)
print(advdf.corr()) # tv > radio > newspaper
# lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data = advdf).fit()
lm_mul = smf.ols(formula='sales ~ tv + radio', data = advdf).fit()
print(lm_mul.summary())

# 예측2 : 새로운 tv, radio 값으로 sales를 추정
x_new2 = pd.DataFrame({'tv':[230.1, 44.5, 100], 'radio':[30.0, 40.0, 50.0]})
pred2 = lm_mul.predict(x_new2)
print('예측 결과2 : ', pred2.values)

print('***' * 30)

# 회귀분석모형의 적절성을 위한 조건 : 아래의 조건 위배 시에는 변수 제거나 조정을 신중히 고려해야 함.
# 1) 정규성 : 독립변수들의 잔차항이 정규분포를 따라야 한다.
# 2) 독립성 : 독립변수들 간의 값이 서로 관련성이 없어야 한다.
# 3) 선형성 : 독립변수의 변화에 따라 종속변수도 변화하나 일정한 패턴을 가지면 좋지 않다.
# 4) 등분산성 : 독립변수들의 오차(잔차)의 분산은 일정해야 한다. 특정한 패턴 없이 고르게 분포되어야 한다.
# 5) 다중공선성 : 독립변수들 간에 강한 상관관계로 인한 문제가 발생하지 않아야 한다.

# 잔차항 구하기
# print(advdf.iloc[:, 0:2])
fitted = lm_mul.predict(advdf.iloc[:, 0:2])
residual = advdf['sales'] - fitted # 잔차 : 표본 데이터의 예측값과 실제값의 차이
print('residual : ', residual[:3])
print(np.mean(residual))

print()
print('선형성 : 독립변수의 변화에 따라 종속변수도 변화하나 특정한 패턴을 가지면 좋지 않다.')