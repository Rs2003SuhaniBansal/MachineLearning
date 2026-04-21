from sklearn.metrics import r2_score


def hypothesis(X, theta): # theta used is transposed (3,1)
    result = []
    for i in range(len(X)):
        summ = 0
        for j in range(len(theta)):
            summ += theta[j][0] * X[i][j]
        result.append([summ])
    return result

def cost_func(X, y, theta):
    hypo = hypothesis(X, theta)
    error_sum = 0
    for i in range(len(hypo)):
        error_sum += (hypo[i][0] - y[i][0])**2
    cost = 1/2 * error_sum
    return cost

def gradient(X, y, theta):
    """performing X transpose"""
    X_transpose = []
    for i in range(len(X[0])):
        row = []
        for j in range(len(X)):
            row.append(X[j][i])
        X_transpose.append(row)
    """generating the error"""
    hypo = hypothesis(X, theta)
    error = []
    for i in range(len(hypo)):
        error.append(hypo[i][0] - y[i][0])
    """generating the gradient"""
    X_T_dot_error = []
    for i in range(len(X_transpose)):
        summ = 0
        for j in range(len(error)):
            summ += X_transpose[i][j] * error[j]
        X_T_dot_error.append([summ])
    return X_T_dot_error

def grad_descent(X, y, theta, alpha, iter):
    for i in range(iter):
        grad = gradient(X, y, theta)
        for j in range(len(X)):
            theta[j][0] = theta[j][0] - alpha * grad[j][0]
            print(theta)
    return theta


    # print(f'{theta:.6f}')

x = [[1, 1, 2], [1, 2, 1], [1, 3, 3]]
y = [[3], [4], [5]]
theta = [[0], [0], [0]] #it is actually theta transpose
# print(hypothesis(x, theta))
# print(cost_func(x, y, theta))
print(gradient(x, y, theta))
theta_updated = grad_descent(x, y, theta, 0.01, 100)
print(f"theta values: \n {theta_updated}")
pred_y = hypothesis(x, theta_updated)
print(f"Predicted value: \n {pred_y}")
r2_score = r2_score(y, pred_y)
print("R2 score:", r2_score)