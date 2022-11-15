# 방법4 : linregress를 사용. model O

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# iq에 따른 시험점수 예측
score_iq = pd.read_csv("../testdata/score_iq.csv")
print(score_iq.head())
print(score_iq.info())

print(score_iq.corr())

x = score_iq.iq
y = score_iq.score

print(np.corrcoef(x, y)[0, 1]) # 0.8822203446134701 설명력? 좋음

# plt.scatter(x, y)
# plt.show()

# 모델 생성
model = stats.linregress(x, y)
print(model)
print('slope : ', model.slope)
print('intercept : ', model.intercept)
print('rvalue : ', model.rvalue)
print('pvalue : ', model.pvalue) # 2.8476895206683644e-50 < 0.05 회귀모델은 유의하다. 두 변수 간에 인과관계가 있다.
print('stderr : ', model.stderr)
# y_hat = 0.6514309527270075 * x + -2.8564471221974657

plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept, c='red')
plt.show()

# 점수 예측
print('점수 예측 : ', model.slope * 140 + model.intercept)
print('점수 예측 : ', model.slope * 125 + model.intercept)
print() # linregress는 predict을 지원하지 않음
new_df = pd.DataFrame({'iq':[140, 125, 123, 100, 95]})
print('점수 예측 : ', np.polyval([model.slope, model.intercept], new_df))

