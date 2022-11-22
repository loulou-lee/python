'''
회귀분석 문제 3) 

kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
회귀분석모형의 적절성을 위한 조건도 체크하시오.
완성된 모델로 Sales를 예측.

"Sales","CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"
'''

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as smf

data = pd.read_csv("../testdata/Carseats.csv")
data['ShelveLoc'] = data['ShelveLoc'].map({'Good':1, 'Bad':0, 'Medium':2})
data['Urban'] = data['Urban'].map({'Yes':1, 'No':0})
data['US'] = data['US'].map({'Yes':1, 'No':0})

result = smf.ols(formula='Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US', data = data).fit()
print(result)
print(result.summary()) # Population Education Urban US 유의 x

lm_mul = smf.ols(formula='Sales ~ CompPrice + Income + Advertising + Price + ShelveLoc + Age + US', data = data).fit()
print(lm_mul)

pred = lm_mul.predict(data[:10])
print('예측값 : ', pred.values)
print('예측값 : ', np.around(pred.values))