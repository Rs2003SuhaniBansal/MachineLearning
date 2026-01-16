import numpy as np

x = np.array([2, 1, 2])
y = np.array([1, 2, 2])

x_transposed = x.T
y_transposed = y.T

dot_prod = np.dot(x_transposed, y_transposed)
print(dot_prod)

#the two vectors give a scalar quantity
#which is the measure of how much they point in the same direction.
#since the above 2 vectors give a positive number they point in the same direction