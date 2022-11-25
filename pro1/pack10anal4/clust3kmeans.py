# 비계층적 군집분석 :
# 군집의 수를 정한 상태에서 설정된 군집의 중심에서 가장 가까운 개체를 하나씩 포함해 나가는 방법으로
# 많은 자료를 빠르고 쉽게 분류할 수 있지만 군집의 수를 미리 정해 줘야 하고 군집을 형성하기 위한
# 초기값에 따라 군집의 결과가 달라진다는 어려움이 있기 때문에 계층적 군집분석을 통해 대략적인 군집의
# 수를 파악하고 이를 초기 군집 수로 설정한다.
# 방법 : K-means Clustering 

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster.tests.test_affinity_propagation import n_clusters
x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
print(x[:3])
# print(y)

# plt.scatter(x[:, 0], x[:, 1], s=50, c='gray', marker='o')
# plt.grid(True)
# plt.show()

from sklearn.cluster import KMeans # 비계층적 군집 분석 중 가장 많이 사용 (데이터의 분포가 비선형인 경우 정확도가 떨어짐)
# 최초의 군집 중심점을 설정하는 방법
# init_centroid = 'random'
init_centroid = 'k-means++' # 기본값 가장 멀리

kmodel = KMeans(n_clusters=3, init=init_centroid, random_state=0)
pred = kmodel.fit_predict(x)
print(pred)
print(x[pred == 0][:3])
print(x[pred == 1][:3])
print(x[pred == 2][:3])
print('centroid(군집 중심점) : ', kmodel.cluster_centers_)

plt.scatter(x[pred == 0, 0], x[pred == 0, 1], s=50, c='red', marker='o')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], s=50, c='green', marker='o')
plt.scatter(x[pred == 2, 0], x[pred == 2, 1], s=50, c='blue', marker='o')
plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:, 1],
            s=50, c='black', marker='+')

def elbow(x):
    sse = []
    for i in range(2, 11):
        km = KMeans().fit(x)

'''
실루엣(silhouette) 기법
  클러스터링의 품질을 정량적으로 계산해 주는 방법이다.
  클러스터의 개수가 최적화되어 있으면 실루엣 계수의 값은 1에 가까운 값이 된다.
  실루엣 기법은 k-means 클러스터링 기법 이외에 다른 클러스터링에도 적용이 가능하다
'''

import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm

# 데이터 X와 X를 임의의 클러스터 개수로 계산한 k-means 결과인 y_km을 인자로 받아 각 클러스터에 속하는 데이터의 실루엣 계수값을 수평 막대 그래프로 그려주는 함수를 작성함.
# y_km의 고유값을 멤버로 하는 numpy 배열을 cluster_labels에 저장. y_km의 고유값 개수는 클러스터의 개수와 동일함.

def plotSilhouette(x, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장
    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기
        c_sil_value = sil_val[pred == c]
        c_sil_value.sort()
        y_ax_upper += len(c_sil_value)

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_sil_value)

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 개수')
    plt.show() 

'''
그래프를 보면 클러스터 1~3 에 속하는 데이터들의 실루엣 계수가 0으로 된 값이 아무것도 없으며, 실루엣 계수의 평균이 0.7 보다 크므로 잘 분류된 결과라 볼 수 있다.
'''
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
km = KMeans(n_clusters=3, random_state=0) 
y_km = km.fit_predict(X)

plotSilhouette(X, y_km)
