# KNN

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target, random_state=66) #계층적 데이터 추출 옵션

# stratify: stratify 파라미터는 분류 문제를 다룰 때 매우 중요하게 활용되는 파라미터 값 입니다. stratify 값으로는 target 값을 지정해주면 됩니다.
# stratify값을 target 값으로 지정해주면 target의 class 비율을 유지 한 채로 데이터 셋을 split 하게 됩니다. 
# 만약 이 옵션을 지정해주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다. 

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

train_acc = []
test_acc = []

neighbors_setting = range(1, 10)

for n_neigh in neighbors_setting:
    clf = KNeighborsClassifier( n_neighbors=n_neigh)
    clf.fit(x_train, y_train)
    train_acc.append(clf.score(x_train, y_train))
    test_acc.append(clf.score(x_test, y_test))
    
import numpy as np
print('train 분류 평균 정확도 : ', np.mean(train_acc))
print(np.mean(test_acc))

plt.plot(neighbors_setting, train_acc, label='train acc')
plt.plot(neighbors_setting, test_acc, label='test acc')
plt.legend()
plt.show()