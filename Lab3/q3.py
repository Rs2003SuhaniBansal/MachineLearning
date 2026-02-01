import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

data = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

x = data.drop(columns=["disease_score_fluct", "disease_score"])
y = data["disease_score"]

X = np.array(x)
Y = np.array(y).reshape(-1, 1)

m = X.shape[0]
X = np.c_[np.ones((m, 1)), X]  #to add x0 for theta0
# print(X)

def hypothesis(X, theta):
    return X.dot(theta)

def cost_func(X, y, theta):
    cost = (1/2) * np.sum((hypothesis(X, theta) - y)  ** 2)
    return cost

def comp_deriv(X, y, theta):
    deriv = X.T.dot(hypothesis(X, theta) - y )
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

    theta_updated = grad_descent(0.000001, X, Y, 1000)
    print(theta_updated)

    y_pred = hypothesis(X, theta_updated)  # predictions
    r2 = r2_score(y, y_pred)
    print("R² Score:", r2)

if __name__ == '__main__':
    main()