from Lab3 import q3
from Lab3 import q1
from Lab2 import q3 as Q3
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"\nData from my implementation for california housing dataset:")

[X, y] = fetch_california_housing(return_X_y=True)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

scaler=StandardScaler()
scaler=scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

def hypothesis(X, theta):
    return X.dot(theta)

def cost_func(X, y, theta):
    m = len(y)
    cost = (1/(2*m)) * np.sum((hypothesis(X, theta) - y)  ** 2)
    return cost

def comp_deriv(X, y, theta):
    m = len(y)
    deriv = (1/m) * X.T.dot(hypothesis(X, theta) - y )
    return deriv

def main():
    def grad_descent(alpha, X, y, iterations):
        n_features = X.shape[1]
        theta = np.zeros((n_features, 1))
        for i in range(iterations):
            grad = comp_deriv(X, y, theta)

            theta = theta - (alpha * grad)

            if i % 200 == 0:
                cost = cost_func(X, y, theta)
                print(f"Iteration {i}, Cost: {cost:.4f}")
        return theta
    theta_updated = grad_descent(0.01, X_train, y_train, 5000)
    print(f"theta values: \n {theta_updated}")

    # y_pred = hypothesis(X_train_scale, theta_updated)  # predictions
    pred_y = hypothesis(X_test, theta_updated)
    print(f"Predicted MedHouseVal value: \n {pred_y}")
    r2 = r2_score(y_test, pred_y)
    print("R² Score:", r2)

if __name__ == '__main__':
    main()

print(f"\nData from scikit learn implementation for california housing dataset:")
California_housing_scikitLearn = Q3.main()
print()
print("_____________________________________________________________________________")
print(f"Data from my implementation for simulated dataset:")
simulated_data_my_implementation = q3.main()
print()
print(f"\nData from scikit learn implementation for simulated dataset:")
simulated_data_scikitLearn = q1.main()