from Lab3 import q3
from Lab3 import q1
from Lab2 import q3 as Q3
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

print(f"\nData from my implementation for california housing dataset:")

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("california_housing_sklearn.csv", index=False)

data = pd.read_csv("california_housing_sklearn.csv")

x = data.drop("MedHouseVal", axis=1).values
y = data["MedHouseVal"].values.reshape(-1, 1)

X = np.array(x)
Y = np.array(y).reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)

m = X.shape[0]
X = np.c_[np.ones((m, 1)), X]  #to add x0 for theta0
# print(X)

def hypothesis(X, theta):
    return X.dot(theta)

def cost_func(X, y, theta):
    m = len(y)
    cost = (1/(2*m)) * np.sum((hypothesis(X, theta) - y)  ** 2)
    return cost

def comp_deriv(X, y, theta):
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

    theta_updated = grad_descent(0.01, X, Y, 3000)
    print(theta_updated)

    y_pred = hypothesis(X, theta_updated)  # predictions
    r2 = r2_score(y, y_pred)
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