'''
[로지스틱 분류분석 문제1]
문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.
'''
# 날씨정보 데이터로 이항분류 : 비가 올지
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # 과적합을 방지하기 위해 data를 쪼갠다
from sklearn.metrics._scorer import accuracy_scorer

data = pd.read_csv("../testdata/dinner_data.csv")
'''
print(data.head(2), data.shape)
data2 = pd.DataFrame()
data2 = data.drop(['요일','소득수준'], axis=1)
data2['외식유무'] = data2['외식유무'].map({'Yes':1, 'No':0}) # Dummy
'''
data2 = data.loc[['토','일']]
print(data2.head(2))
print(data2['외식유무'].unique()) # [1 0]
# RainTomorrow : 종송변수, 그 외는 독립변수
formula = '외식유무 ~ 요일 + 소득수준'
print(formula)
result = smf.logit(formula=formula, data=data2).fit()
print(result.summary())
# print(model.params)

pred = result.predict(data2[:10])
print('예측값 : ', pred.values)
print('예측값 : ', np.around(pred.values)) # np.around() 0.5를 기준으로 0, 1로 출력
print('실제값 : ', data2['외식유무'][:10].values)
# y=w * 2 + 0 수학에서는 100%의 답을 원함
# 통계에서는 4의 주변값이 나올 수 있도록 학습을 함.
# 예를 들어 개 이미지 분류를 하는 경우 꼬리가 없는 개도 정확하게 분류되도록 하는 것이 머신러닝의 목적
# 표용성이 있는 모델이라 함은 데이터 분류 인식률이 80%, 90% 등인 것이 100% 인 경우보다 더 효과적이다.
