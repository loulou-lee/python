# 선형회귀분석 : iris dataset으로 모델 생성
# 약한 상관관계 변수, 약한 상관관계 변수로 모델 작성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sms

iris = sns.load_dataset('iris')
print(iris.head(3))
print(type(iris))
print(iris.corr())

print('연습1 : 약한 상관관계 변수 - sepal_length, sepal_width')
result1 = sms.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('요약결과1 : ', result1.summary())
print('R-squared : ', result1.rsquared) # 설명력이 너무 낮다.
print('p-value : ', result1.pvalues[1]) # > 0.05이므로 독립변수로 유의하지 않다
# 의미없는 모델로 예측 결과 확인
print('실제값 : ', iris.sepal_length[:5].values)
print('예측값 : ', result1.predict()[:5])

# model1 시각화
# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color='r')
# plt.show()

print('\n연습2 : 강한 상관관계 변수 - sepal_length, petal_width')
result2 = sms.ols(formula='sepal_length ~ petal_width', data=iris).fit() # 왜 smf로 햇지?
print('요약결과2 : ', result2.summary())
print('R-squared : ', result2.rsquared) # 설명력이 너무 낮다.
print('p-value : ', result2.pvalues[1]) # > 0.05이므로 독립변수로 유의하지 않다
# 의미없는 모델로 예측 결과 확인
print('실제값 : ', iris.sepal_length[:5].values)
print('예측값 : ', result2.predict()[:5])

# model2 시각화
plt.scatter(iris.petal_width, iris.sepal_length)
plt.plot(iris.petal_width, result2.predict(), color='b')
plt.show()

print('새로운 값(petal_width)으로 결과 예측(sepal_length)')
new_data = pd.DataFrame({'petal_width':[1.1, 3.3, 5.5, 7.7]})
y_pred = result2.predict(new_data)
print('예측결과 : ', y_pred.values)