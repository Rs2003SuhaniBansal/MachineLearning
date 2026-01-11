import numpy as np

def y(x1, x2, x3):
    return 2*x1 + 3*x2 + 3*x3 + 4

def grad(x1, x2, x3):
    return np.array([2, 3, 3])

points = [[2, 3, 5],
          [0, -1, 2],
          [1, 0, 9]]

for i in points:
    x1, x2, x3 = i
    g = grad(x1, x2, x3)
    val = y(x1, x2, x3)
    print(f"At point {i}: y = {val}, gradient = {g}")