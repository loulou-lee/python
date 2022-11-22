# ROC(Reciever Operating C) curve
# ROC curve는 모든 가능한 threshold에 대해 분류모델의 성능을 평가하는 데 사용된다.
# ROC curve 아래의 영역은 AUC이다.

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
print(x[:3])
print(y[:3])

import matplotlib.pyplot as plt
plt.scatter(x[:, 0], x[:, 1])
plt.show()

model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print('y_hat : ', y_hat[:3])

print()
f_value = model.decision_function(x) # 판별함수(결정함수) : 판별 경계선 설정을 위한 샘플 얻기
# print('f_value : ', f_value)

df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f', 'y_hat', 'y'])
print(df.head(3))

print()
print(confusion_matrix(y, y_hat))
acc = (44 + 44) / 100 # (TP + TN) / 전체수
recall = 44 / (44 + 4) # TP / (TP + FN)
precision = 44 / (44 + 8) # TP / (TP + FP)
specificity = 44 / (8 + 44) # TN / (FP + TN)
fallout = 8 / (8 + 44) # FP / (FP + TN)

print('acc(정확도) : ', acc)
print('recall(재현율) : ', recall)    # TPR
print('precision(정밀도) : ', precision)
print('specificity(특이도) : ', specificity)
print('fallout(위양성률) : ', fallout) # FPR
print('fallout(위양성률) : ', 1 - specificity)

print()
from sklearn import metrics
ac_sco = metrics.accuracy_score(y, y_hat)
print('ac_sco : ', ac_sco)

cl_rep = metrics.classification_report(y, y_hat)
print('cl_rep : ', cl_rep)

print()
fpr, tpr, threshold = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류결정 임계값(positive 예측값을 결정하는 확률 기준값) : ', threshold)

# ROC 커브
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='random classifier line(AUC:0.5)')
plt.plot([fallout], [recall], 'ro', ms=10) # 위양성율, 재현율
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

print('AUC : ', metrics.auc(fpr, tpr)) # 1에 근사할 수록 좋은 분류모델
plt.show()