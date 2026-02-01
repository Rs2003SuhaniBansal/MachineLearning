import pandas as pd
import numpy as np
from Lab3 import q2

data = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

x = data.drop(columns=["disease_score_fluct"])
y = data["disease_score_fluct"]

X = np.array(x)
Y = np.array(y).reshape(-1, 1)
m = X.shape[0]
X = np.c_[np.ones((m, 1)), X]

def normal_eqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta
theta = normal_eqn(X, Y)
print(theta)

def hypothesis(X, theta):
    return X.dot(theta)

y_pred = hypothesis(X, theta)

from sklearn.metrics import r2_score

r2 = r2_score(y, y_pred)
print("R² (Normal Equation):", r2)

print(f"\nData from scikit learn implementation for simulated dataset:")
simulated_data_scikitLearn = q2.main()

