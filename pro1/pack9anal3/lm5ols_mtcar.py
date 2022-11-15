# mtcars dataset으로 단순/다중회귀 모델 작성 : ols() 사용
# 회귀선(추세선)을 그리기 위해 추세식을 계산하는 함수 polyfit를 사용
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3)) # 32 rows x 11 columns
# print(mtcars.corr())
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0,1])
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0,1])

# 단순선형회귀 : mtcars.hp(feature, x), mtcars.mpg(label, y)
# 시각화
"""
plt.scatter(mtcars.hp, mtcars.mpg)
# 참고 : numpy의 polyfit()을 이용하면 slope, intercept를 얻을 수 있음
slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1)
print('slope:{}, intercept:{}'.format(slope, intercept))
plt.plot(mtcars.hp, slope*mtcars.hp + intercept)
plt.xlabel('마력수')
plt.ylabel('연비')
plt.show()
"""
result1 = smf.ols('mpg ~ hp', data=mtcars).fit()
print(result1.summary())
print(result1.conf_int(alpha=0.05))
print()
print(result1.summary().tables[1])

print('마력수 110에 대한 연비는', -0.088895*110 + 30.0989)
print('마력수 50에 대한 연비는', -0.088895*50 + 30.0989)
print('마력수 200에 대한 연비는', -0.088895*200 + 30.0989)

print('---------------')
# 다중선형회귀 : mtcars.hp, mtcars.wt(feature, x), mtcars.mpg(label, y)
result2 = smf.ols('mpg ~ hp + wt', data=mtcars).fit() # 독립변수가 복수일때는 수정된 r-squared 보기
print(result2.summary())
print(result2.summary().tables[1])
print('마력수 110, 차체 무게 5톤에 대한 연비는 ', (-0.0318 * 110) + (-3.8778 * 5) + 37.2273)

print('predict 함수 사용')
new_data = pd.DataFrame({'hp':[110, 120, 150], 'wt':[5, 2, 7]})
new_pred = result2.predict(new_data)
print('예상 연비 : ', new_pred.values)

# 키보드로 값 받기
new_hp = float(input('새로운 마력수 : '))
new_wt = float(input('새로운 차체무게 : '))
new_data2 = pd.DataFrame({'hp':[new_hp], 'wt':[new_wt]})
new_pred2 = result2.predict(new_data2)
print('예상 연비 : ', new_pred2.values)
