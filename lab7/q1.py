from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    sonar = fetch_openml(name='sonar', version=1, as_frame=False)
    """as_frame=True will load data as pd.dataframe and 
    False will run it as a numpy array which is used later in the code.
     Therefore False is used, won't run with as_frame=True"""

    X = sonar.data
    y = sonar.target
    # print(X.head())

    k = 10  # K-fold
    samples = len(X)
    idx = np.arange(samples)
    np.random.shuffle(idx)

    """getting the length of the train and test sets"""
    test_size = int(samples / k)
    train_size = int(samples - test_size)

    accuracy_list = []
    for fold in range(k):
        start = fold * test_size
        end = start + test_size
        test_set = idx[start:end]
        train_set = np.concatenate((idx[:start], idx[end:]))

        X_test = X[test_set]
        X_train = X[train_set]
        y_train = y[train_set]
        y_test = y[test_set]

        scaler_obj = StandardScaler()
        scaler_parameter = scaler_obj.fit(X_train)
        X_train = scaler_obj.transform(X_train, scaler_parameter)
        X_test = scaler_obj.transform(X_test, scaler_parameter)

        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred_y)
        accuracy_list.append(accuracy)

    print("list of accuracies:",accuracy_list)

    test_accuracy = np.mean(accuracy_list)
    std_dev = np.std(accuracy_list)

    print("Standard Deviation:", std_dev)
    print("Test Accuracy:", test_accuracy)


if __name__ == '__main__':
    main()

