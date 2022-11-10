# 공분산 / 상관계수

import numpy as np
import matplotlib.pyplot as plt

#  공분산 예
print(np.cov(np.arange(1, 6), np.arange(2, 7))) # 2.5
print(np.cov(np.arange(10, 60, 10), np.arange(20, 70, 10))) # 250
print(np.cov(np.arange(1, 6), (3,3,3,3,3))) # 0
print(np.cov(np.arange(1, 6), np.arange(6, 1, -1))) # -2.5

#공분산을 표준화한게 상관계수이다
print()
x = [8, 3, 6, 6, 9, 4, 3, 9, 3, 4]
print('x 평균 : ', np.mean(x))
print('x 분산 : ', np.var(x))

y = [60, 20, 40, 60, 90, 50, 10, 80, 40, 50]
print('y 평균 : ', np.mean(y))
print('y 분산 : ', np.var(y))

# plt.scatter(x, y)
# plt.show()

print('x, y 공분산 : ', np.cov(x, y)[0, 1])
print('x, y 상관계수 : ', np.corrcoef(x, y)[0, 1]) # x, y 상관계수 :  0.8663686463212853

from scipy import stats
print(stats.pearsonr(x, y))
print(stats.spearmanr(x, y))

# 주의 : 공분산이나 상관계수는 선형 데이터인 경우에 활용
m = [-3, -2, -1, 0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]
plt.scatter(m, n)
plt.show()
print('m, n 공분산 : ', np.cov(m, n)[0, 1])
print('m, n 상관계수 : ', np.corrcoef(m, n)[0, 1])

