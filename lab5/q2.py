import numpy as np
import matplotlib.pyplot as plt

"""a range of values of z is defined for the sigmoid function"""
z = np.linspace(-10, 10, 300)

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))
y = sigmoid_function(z)

# Plot the sigmoid curve
plt.figure()
plt.plot(z, y)
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Function")
plt.show()