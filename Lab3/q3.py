import pandas as pd
import numpy as np

data = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

x = data.drop(columns=["disease_score_fluct"])
y = data["disease_score_fluct"]

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
    def grad_descent(theta, alpha, X, y, iterations):
        for i in range(iterations):
            c = cost_func(X, y, theta)
            grad = comp_deriv(X, y, theta)
            theta = theta - alpha * grad
            print(f" iteration no.: {i}, theta: {theta}, cost: {c}")
            if np.isinf(c):
                break

    grad_descent(theta, learning_rate, X, Y, iterations)

n_features = X.shape[1]
theta = np.zeros((n_features, 1))
learning_rate = 0.00000001
iterations = 1000

if __name__ == '__main__':
    main()