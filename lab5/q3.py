import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

"""calling the sigmoid function for calculating the derivation"""
def deriv_sigmoid(z):
    return sigmoid(z) * (1-sigmoid(z))

z = np.linspace(-10, 10, 300)
y = deriv_sigmoid(z)
print(y)

plt.figure()
plt.plot(z, y)
plt.title("Derivative of Sigmoid Function")
plt.show()