'''
회귀분석 문제 2) 
testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
  - 국어 점수를 입력하면 수학 점수 예측
  - 국어, 영어 점수를 입력하면 수학 점수 예측
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

df = pd.read_csv("../testdata/student.csv")
print(df.head(3))
print(df.corr())

result1 = smf.ols('수학 ~ 국어', data=df).fit()
print(result1.summary())
print(result1.conf_int(alpha=0.05))
print()
print(result1.summary().tables[1])
# 키보드로 값 받기
kor = float(input('국어점수 : '))
print('수학 예상 점수 : ', 0.5705*kor+32.1069)

print('---------------')
result2 = smf.ols('수학 ~ 국어 + 영어', data=df).fit() # 독립변수가 복수일때는 수정된 r-squared 보기
print(result2.summary())
print(result2.summary().tables[1])

# 키보드로 값 받기
korean = float(input('국어점수 : '))
english = float(input('영어점수 : '))
new_data2 = pd.DataFrame({'국어':[korean], '영어':[english]})
new_pred2 = result2.predict(new_data2)
print('예상 점수 : ', new_pred2.values)
