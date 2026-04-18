import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

red = np.array([[3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]])
blue = np.array([[1,13],[1,18],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1]])

y = np.array([["Blue"],["Blue"],["Blue"],["Blue"],["Blue"],["Blue"],["Blue"],["Blue"]
    ,["Red"],["Red"],["Red"],["Red"],["Red"],["Red"],["Red"],["Red"]])

plt.scatter(blue[:,0], blue[:,1])
plt.scatter(red[:,0], red[:,1])
plt.show()

def Transform(pt):
    x1 = pt[:,0]
    x2 = pt[:,1]
    z1 = x1**2
    z2 = 2**0.5 * x1 * x2
    z3 = x2**2
    return np.column_stack((z1,z2,z3))

red_3d = Transform(red)
blue_3d = Transform(blue)

w1, w2, w3 = 0.5, -0.5, 0.5
b = -50
z1_range = np.linspace(0, 400, 20)
z2_range = np.linspace(0, 400, 20)

Z1, Z2 = np.meshgrid(z1_range, z2_range)

# Solve plane equation for Z3
Z3 = (-w1*Z1 - w2*Z2 - b) / w3

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(blue_3d[:,0], blue_3d[:,1], blue_3d[:,2], label='Blue')
ax.scatter(red_3d[:,0], red_3d[:,1], red_3d[:,2], label='Red')
ax.plot_surface(Z1, Z2, Z3, alpha=0.3)

ax.set_xlabel('x1^2')
ax.set_ylabel('√2 x1x2')
ax.set_zlabel('x2^2')

plt.legend()
plt.show()