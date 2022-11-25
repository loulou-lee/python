# MLP : breast_cancer dataset

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())

x = cancer['data']
y = cancer['target']
print(cancer.target_names) # ['malignant' 'benign']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train, x_test)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), solver='adam', 
                    max_iter=100, learning_rate_init=0.1, verbose=1, random_state=1).fit(x_train, y_train)
pred = mlp.predict(x_test)
print(pred[:5])
print(y_test[:5])
print(mlp.score(x_train, y_train))
print(mlp.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))