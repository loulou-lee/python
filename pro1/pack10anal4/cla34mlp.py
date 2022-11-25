# MLP(다층 신경망)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
label = np.array([0,0,0,1]) # and

# model = MLPClassifier(hidden_layer_sizes=10, solver='adam', learning_rate_init=0.01).fit(feature, label)
# model = MLPClassifier(hidden_layer_sizes=10, solver='adam', learning_rate_init=0.01,
#                       max_iter=10, verbose=1).fit(feature, label)

model = MLPClassifier(hidden_layer_sizes=(10,10,10), solver='adam', learning_rate_init=0.1,
                      max_iter=100, verbose=1).fit(feature, label)

pred = model.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))