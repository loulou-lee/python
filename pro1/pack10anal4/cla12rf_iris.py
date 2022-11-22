# iris dataset을 사용 - 꽃의 종류를 세가지로 분류한다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn import datasets

iris = datasets.load_iris()
# print(iris.DESCR)
print(iris.keys())
x = iris.data
print(x)
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])) # 0.96286543

x = iris.data[:, [2, 3]] # petal.length, petal.width만 참여
y = iris.target
print(x[:3])
print(y[:3], ' ', set(y)) # {0, 1, 2}

# train / test split (7 : 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
'''
# data scaling : 표준화 - 최적화 과정에서  안전성, 수렴 속도 향상, 오버/언더 플로우 방지 ... 가능
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

# model = LogisticRegression(C=1.0, random_state = 0, verbose=0) # C=1.0 L2규제(패널티 적용) - 값이 작을수록 규제는 강화됨
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1, n_jobs=1)
print(model)
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))

# 분류 정확도(accuracy) 1
print('%.5f'%accuracy_score(y_test, y_pred))

# 분류 정확도(accuracy) 2
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

# 분류 정확도(accuracy) 3
print('test : ', model.score(x_test, y_test))
print('train : ', model.score(x_train, y_train))

# 모델 저장
import pickle
pickle.dump(model, open('cla_model.sav', mode='wb'))
del model

mymodel = pickle.load(open('cla_model.sav', mode='rb'))

print('새로운 값으로 분류 예측 - petal.length, petal.width만 참여')
print(x_test[:1])
new_data = np.array([[5.1, 2.4], [0.3, 0.3] [3.4, 0.2]])
# 참고 : 만약 표준화로 학습했다면 new_data도 표준화 해 줘야 한다.
new_pred = mymodel.predict(new_data)
print('예측 결과 : ', new_pred)
print(mymodel.predict_proba(new_data))

# * 붓꽃 자료에 대한 로지스틱 회귀 결과를 차트로 그리기 *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager, rc

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
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, 
                     test_idx=range(105, 150), title='scikit-learn제공')
