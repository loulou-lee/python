# iris dataset으로 지도학습(KNN) 과 비지도학습(KMeans) 실행

from sklearn.datasets import load_iris

iris_dataset=load_iris()

print(iris_dataset['data'][:3])
print(iris_dataset['feature_names'])

#train/test split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_dataset['data'],iris_dataset['target'],
                                                    test_size=0.25, random_state=42)

# 지도학습(KNN)
from sklearn.neighbors import KNeighborsClassifier
knnModel =KNeighborsClassifier(n_neighbors=5) #metric='euclidean'
knnModel.fit(train_x, train_y) #feature와 label 다줌

predict_label = knnModel.predict(test_x)
print(predict_label)

from sklearn import metrics
print('acc : ', metrics.accuracy_score(test_y, predict_label))

print()
# 비지도학습(K-Means)
from sklearn.cluster import KMeans
#knnModel 비교할려고 KmeansModel이라고 이름붙임
KmeansModel = KMeans(n_clusters=3, init='k-means++', random_state=0)
KmeansModel.fit(train_x) #label안줌

print(KmeansModel.labels_)
print('0 cluster : ', train_y[KmeansModel.labels_ == 0])
print('1 cluster : ', train_y[KmeansModel.labels_ == 1])
print('2 cluster : ', train_y[KmeansModel.labels_ == 2])

pred_cluster = KmeansModel.predict(test_x)
print('pred_cluster : ',pred_cluster)

import numpy as np
#어레이로 변환
np_arr = np.array(pred_cluster)

# np_arr[np_arr == 3] = 1
# np_arr[np_arr == 3] = 0
# np_arr[np_arr == 3] = 2
pred_label = np_arr.tolist()
print(pred_label)
print('test acc : {:.2f}'.format(np.mean(pred_label == test_y)))