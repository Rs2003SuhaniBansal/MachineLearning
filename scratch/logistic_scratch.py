import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,0],
              [2,1],
              [3,1],
              [4,2],
              [5,3]])

y = np.array([0,0,0,1,1]).reshape(-1,1)

# Add bias
X = np.c_[np.ones(X.shape[0]), X]

def sigmoid(z):
    return 1/(1+np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(X @ theta)

def gradient(X,y,theta):
    h = hypothesis(X,theta)
    grad = X.T @ (h - y)
    return grad

def train(X,y,alpha,iterations):
    theta = np.zeros((X.shape[1],1))

    for i in range(iterations):
        grad = gradient(X,y,theta)
        theta = theta - alpha * grad

    return theta

theta = train(X,y,0.01,1000)

print("Theta values:\n",theta)

pred = hypothesis(X,theta)

print("Predictions:\n",pred)

plt.plot(pred,marker='o',label="Sigmoid output")
plt.plot(y,marker='x',linestyle='--',label="True labels")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.title("Sigmoid Predictions vs True Labels")
plt.legend()
plt.show()