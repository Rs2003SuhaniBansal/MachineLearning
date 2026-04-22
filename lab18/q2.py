import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_iris()
x = data.data
Y = data.target

def main():

    classes = Y < 2
    X = x[classes, :2]
    y = Y[classes]

    np.random.seed(10)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:90], X[90:]
    y_train, y_test = y[:90], y[90:]

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()