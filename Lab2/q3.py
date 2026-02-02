from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
def main():

    #Loading the data
    [X, y] = fetch_california_housing(return_X_y=True)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    #Standardizing the data
    scaler=StandardScaler()
    scaler=scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    #initializing the model
    model = LinearRegression()

    # training a model
    model.fit(X_train, y_train)

    pred_y=model.predict(X_test)
    print(f"Predicted MedHouseVal value: \n {pred_y}")

    # print r2 score
    r2 = r2_score(y_test, pred_y)
    print("R² Score:", r2)
    theta = model.coef_
    print("theta values:",theta)

if __name__ == '__main__':
    main()
