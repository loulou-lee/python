# 단순선형회귀 모델
# 기본적인 결정론적 선형회귀 방법 : 독립변수에 대해 대응하는 종속변수와 유사한 예측값을 출력하는 함수 f(x)를 찾는 작업이다.
import pandas as pd

df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head(3))
print(df.corr())

import statsmodels.formula.api as smf

# 적절성이 만족도에 영향을 준다라는 가정하에 모델 생성
model = smf.ols(formula = '만족도 ~ 적절성', data=df).fit() # 모델의 파라미터가 성능을 결정한다
# R에서는 fit()이 내부적으로 처리된다
print(model.summary()) # 생성된 모델의 요약결과를 반환. 능력치를 확인.
# 잔차의 독립성 : Durbin-Watson:                   2.185

print(' 회귀계수 : ', model.params) # 기울기와 y절편이 회귀 계수이다
print(' 결정계수 : ', model.rsquared)
print(' 유의확률 : ', model.pvalues)
print(' 예측값 : ', model.predict()[:5])
print(' 실제값 : ', df.만족도[:5].values)

print()
new_df = pd.DataFrame({'적절성':[4,3,2,1]})
print(new_df)
new_pred = model.predict(new_df)
print('예측 결과 : ', new_pred)