"""L1 and L2 norm from scratch"""
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

"""Adding l2 norm in the cost function"""
def cost_func(X, y, theta, l):
    cost = (1/2) * np.sum((hypothesis(X, theta) - y)  ** 2) + l*(np.sum(np.square(theta[1:])))
    return cost

"""adding l2 norm while computing the derivative or the gradient"""
def comp_deriv(X, y, theta, l):
    deriv = X.T.dot(hypothesis(X, theta) - y )

    l2_grad = 2 * l * theta
    l2_grad[0] = 0
    return deriv + l2_grad

def main():
    def grad_descent(alpha, X, y, iterations, l):
        n_features = X.shape[1]
        theta = np.zeros((n_features, 1))
        for i in range(iterations):
            grad = comp_deriv(X, y, theta, l)

            theta = theta - (alpha * grad)

            if i % 200 == 0:
                cost = cost_func(X, y, theta, l)
                print(f"Iteration {i}, Cost: {cost:.4f}")
        return theta

    theta_updated = grad_descent(0.000001, X, Y, 1000, 0.1)
    print(f"theta values: \n{theta_updated}")

    y_pred = hypothesis(X, theta_updated)  # predictions
    r2 = r2_score(y, y_pred)
    print("R² Score:", r2)

if __name__ == '__main__':
    main()