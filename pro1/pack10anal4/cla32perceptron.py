# Perceptron(퍼셉트론, 단층신경망)이 학습할 때 주어진 데이터를 학습하고 에러가 발생한 데이터에 기반하여 Weight(가중치)값을 기존에서
# 새로운 W값으로 업데이트 시켜주면서 학습. input의 가중치합에 대해 입계값을 기준으로 두 가지 output 중 한 가지를 출력하는 구조.

# 논리회로로 실습
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics._scorer import accuracy_scorer

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
label = np.array([0,0,0,1]) # and

ml = Perceptron(max_iter=1, eta0=0.1, verbose=1).fit(feature, label) # verbose=0 학습내용 안보여줌 verbose=1, 학습내용 보여줌
# max_iter=1000 == 천번을 학습하라
# eta0 학습률
# loss랑 accuracy 반비례
print(ml)
pred = ml.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))

