# 특성공학 기법 중 차원축소(PCA - 주성분 분석)
# n개의 관측치와 p개의 변수로 구성된 데이터를 상관관계가 최소화된 k개의 변수로 축소된 데이터를 만든다.
# 데이터의 분산을 최대한 보존하는 새로운 축을 찾고 그 축에 데이터를 사영시키는 기법. 직교
# 목적 : 독립변수(x, feature)의 갯수를 줄임. 이미지 차원 축소로 용량을 최소화.

# iris dataset으로 PCA를 진행
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
n = 10
x = iris.data[:n, :2]
print(x, x.shape, type(x))
print(x.T)

# plt.plot(x.T, 'o:')
# plt.xticks(range(2))
# plt.grid()
# plt.legend(['표본{}'.format(i) for i in range(n)])
# plt.show()

'''
# 산점도
df = pd.DataFrame(x)
# print(df)
ax = sns.scatterplot(0, 1, data=pd.DataFrame(x), marker='s', s=100, color='.2')
for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] - 0.05, '표본{}'.format(i + 1))
plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침너비')
plt.axis('equal')
plt.show()
'''

# PCA
pca1 = PCA(n_components=1) # n_components=1 변환할 차원수 1
x_low = pca1.fit_transform(x) # 비지도학습. 차원 축소된 근사 데이터
print('x_low : ', x_low, ' ', x_low.shape)

x2 = pca1.inverse_transform(x_low) # 차원 축소된 근사 데이터를 원복
print('원복된 결과 : ', x2, ' ', x2.shape)
print(x)
print(x_low[0])
print(x2[0, :])
print(x[0])
'''
ax = sns.scatterplot(0, 1, data=pd.DataFrame(x), marker='s', s=100, color='.2')
for i in range(n):
    d = 0.03 if x[i, 1] > x2[i, 1] else - 0.04
    ax.text(x[i, 0] - 0.05, x[i, 1] - d, '표본{}'.format(i + 1))
    plt.plot([x[i, 0], x2[i, 0]], [x[i, 1], x2[i, 1]], 'k--')

plt.plot(x2[:, 0], x2[:, 1], 'o-', color='b', markersize=10)    
plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침너비')
plt.axis('equal')
plt.show()
'''

# iris 4개의 열을 모두 참여
x = iris.data
pca2 = PCA(n_components = 2)
x_low2 = pca2.fit_transform(x)
print('x_low2 : ', x_low2[:3], ' ', x_low2.shape)
print(pca2.explained_variance_ratio_) # 전체 변동성에서 개별 PCA 결과(개별 component) 별로 차지하는 변동성 비율을 제공
# [0.92461872 0.05306648]
x4 = pca2.inverse_transform(x_low2)
print('최초 자료 : ', x[0]) # 최초 자료 :  [5.1 3.5 1.4 0.2]
print('차원 축소 : ', x_low2[0]) # 차원 축소 :  [-2.68412563  0.31939725]
print('차원 복귀 : ', x4[0]) # 차원 복귀 :  [5.08303897 3.51741393 1.40321372 0.21353169]
# PCA를 통해 근사행렬로 변환된다

print()
iris2 = pd.DataFrame(x_low2, columns=['f1', 'f2'])
print(iris2.head(3))

