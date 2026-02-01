import numpy as np
def hypothesis(X, theta):
    return X.dot(theta)

def cost_func(X, y, theta):
    cost = (1/2) * np.sum((hypothesis(X, theta) - y)  ** 2)
    return cost

def comp_deriv(X, y, theta):
    deriv = X.T.dot(hypothesis(X, theta) - y )
    return deriv

def grad_descent(theta, alpha, X, y, iterations):
    for i in range(iterations):
        c = cost_func(X, y, theta)
        grad = comp_deriv(X, y, theta)
        theta = theta - alpha * grad
        print(f" iteration no.: {i}, theta: {theta}, cost: {c}")