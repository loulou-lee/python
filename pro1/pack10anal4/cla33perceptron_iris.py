# Logistic Regression : 다항분류 (활성화 함수 softmax : 결과값을 확률로 변환) # 이항분류인 시그모이드를 쓸 수 없다.)
# Logistic Regression은 다중 클래스(label-범주형)를 지원하도록 일반화 되어 있다. softmax 함수를 사용하기 때문이다.
# iris dataset을 사용 : 꽃의 종류를 세 가지로 분류한다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # feature 표준화

iris = datasets.load_iris() # dataframe이 아니라 load 해도 내용을 바로 보여주지 않는다.
# print(iris.DESCR)
print(iris.keys())
# x = iris.data
# print(x)

''' 이렇게 가져와도 된다.
data = pd.read_csv('../testdata/iris.csv')
print(data.head(3))
#    Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
'''

print()
# 상관관계 확인
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])) 
# 0.96286543 # Petal.Length-Petal.Width 양의 상관관계 높다.

print()
x = iris.data[:, [2,3]] # Petal.Length와 Petal.Width만 참여
y = iris.target
print(x[:3])
print(y[:3], ' ', set(y)) # 중복 허용 x (unique 기능) # 종류 {0, 1, 2} 세 가지

print()
# train / test split (7:3)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,) # 150개가 105개, 45개로 쪼개짐 # 열은 상관관계 강한 2개만

'''
print()
# data scaling : 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버/언더 플로우 방지 ... 가능
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])

# 스케일링 자료 원복
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
'''

print()
# model = LogisticRegression(C=1.0, random_state=0, verbose=0) # C=1.0 L2규제 (패널티 적용) - 값이 작을 수록 규제는 강화된다.
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1, n_jobs=1)

# from sklearn import svm
# model = svm.SVC(C=1)
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

from sklearn.linear_model import Perceptron
model = Perceptron(max_iter=100, eta0=0.1, random_state=1)
print(model)
model.fit(x_train, y_train)

print()
# 분류 예측
y_pred = model.predict(x_test)
print('예측값 :', y_pred)
print('실제값 :', y_test)
print('총 개수 :%d, 오류 수 :%d'%(len(y_test), (y_test != y_pred).sum()))

print()
# 분류 정확도 accuracy 1
print('%.5f'%accuracy_score(y_test, y_pred)) # iris의 맥시멈

print()
# # 분류 정확도 accuracy 2
# con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
# print(con_mat)
# print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test)) # 맞은 것만

print()
# 분류 정확도 accuracy 3
print('test :', model.score(x_test, y_test))
print('train :', model.score(x_train, y_train))

print()
# 모델 저장
import pickle
pickle.dump(model, open('cla_model.sav', mode='wb'))
del model

mymodel = pickle.load(open('cla_model.sav', mode='rb'))

print()
# 새로운 값으로 분류 예측 - petal.length, petal.width만 참여
print(x_test[:1]) # 첫 데이터 [[5.1, 2.4]] 사용
new_data = np.array([[5.1, 2.4], [0.3, 0.3], [3.4, 0.2]])
# 참고 : 만약 표준화로 학습했다면 new_data도 표준화해야 한다.
new_pred = mymodel.predict(new_data) # softmax가 반환한 결과 중 가장 큰 인덱스를 취한 결과
print('예측 결과 :', new_pred)
# print(mymodel.predict_proba(new_data)) # 확률값으로 보기 # 내부적으로 softmax가 반환한 결과


print()
print()
# 시각화
# 붓꽃 자료에 대한 로지스틱 회귀 결과를 차트로 그리기

from matplotlib.colors import ListedColormap
plt.rc('font', family='malgun gothic')      
plt.rcParams['axes.unicode_minus']= False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, test_idx=range(105, 150), title='scikit-learn제공')    
