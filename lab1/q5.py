import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-10, 10, 100)
y = x1**2

plt.figure()
plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = x1^2")
plt.show()

x1_points = [-5, -3, 0, 3, 5]
for i in x1_points:
    deriv = 2*i
    print(f"For x1 = {i}, deriv = {deriv}")

print("for value of x1 at which y = 0 is x1 = 0")

