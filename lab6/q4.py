from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

def main():
    [X,y] = fetch_california_housing(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=999)

    scaler_obj = StandardScaler()
    scaler_parameter = scaler_obj.fit(X_train)

    X_train = scaler_obj.transform(X_train, scaler_parameter)
    X_test = scaler_obj.transform(X_test, scaler_parameter)
    X_val = scaler_obj.transform(X_val, scaler_parameter)

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    model = LinearRegression()
    model.fit(X_train, y_train)

    #Model Validation
    pred_y = model.predict(X_test)
    print("normal prediction:", pred_y)
    r2 = r2_score(y_test, pred_y)
    print("R² of normal pred:", r2)

    val_pred = model.predict(X_val)
    print("validation prediction: ", val_pred)
    val_r2 = r2_score(y_val, val_pred)
    print("Validation R²:", val_r2)

if __name__ == '__main__':
    main()

