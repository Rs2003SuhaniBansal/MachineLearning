"""stochastic gradient descent"""
import numpy as np
from sklearn.metrics import r2_score

X = np.array([[1, 1, 2], [1, 2, 1], [1, 3, 3]])
y = np.array([[3], [4], [5]])
# print(X.shape)

"""no changes will be made to the hypothesis since this returns an array with all theta values 
for all the samples therefore the number of theta values is equal to the number of samples
but in stochastic gradient descent only 1 sample is taken, so it should generate only 1 value
and hence it's shape will be (1,1). """
def hypothesis(X, theta): # theta used is transposed (3,1)
    return X.dot(theta)

"""np.sum is not required since only 1 sample is there"""
def cost_func(X, y, theta):
    cost = 1/2 * ((hypothesis(X, theta) - y)  ** 2)
    return cost

"""for 1 sample summation of gradient for all samples does not occur."""
def comp_deriv(X, y, theta): #gradient
    return X.T * (hypothesis(X, theta) - y)

"""here the random.randint will generate a random sample for both the features 
    and the true value (y) which is then reshaped to restore the dimensions
    this is put inside the iteration loop so that again a random sample is generated.
    this ensures that the predicted values of y remain constant."""
def main():
    def grad_descent(alpha, X, y, iterations):
        theta = np.zeros((X.shape[1], 1))
        for i in range(iterations):
            a = np.random.randint(0, X.shape[0])
            x_i = X[a].reshape(1, -1)
            y_i = y[a].reshape(1, 1)

            grad = comp_deriv(x_i, y_i, theta)
            # hyp = hypothesis(x_i, theta)
            # print(hyp.shape)

            theta = theta - (alpha * grad)

            if i % 200 == 0:
                cost = cost_func(x_i, y_i, theta)
                print(f"Iteration {i}, Cost: {cost}")
        return theta
    theta_updated = grad_descent(0.01, X, y, 1000)
    print(f"theta/coefficient values for all features: \n {theta_updated}")

    # y_pred = hypothesis(X_train_scale, theta_updated)  # predictions
    pred_y = hypothesis(X, theta_updated)
    print(f"Predicted values for all samples: \n {pred_y}")
    r2 = r2_score(y, pred_y)
    print("R² Score:", r2)

if __name__ == '__main__':
    main()