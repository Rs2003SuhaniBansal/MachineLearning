import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    [X, y] = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    scaler_obj = StandardScaler()
    scaler_parameter = scaler_obj.fit(X_train)
    X_train = scaler_obj.transform(X_train, scaler_parameter)
    X_test = scaler_obj.transform(X_test, scaler_parameter)

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # initializing the model
    """using the logistic regression model from scikit learn"""
    model = LogisticRegression()

    # training a model
    model.fit(X_train, y_train)

    pred_y = model.predict(X_test)
    print(f"Predicted MedHouseVal value: \n {pred_y}")

    """Checking the accuracy of the predicted values by the model since it is a classification
    problem and prediction doesn't have numerical values."""
    accuracy = accuracy_score(y_test, pred_y)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
