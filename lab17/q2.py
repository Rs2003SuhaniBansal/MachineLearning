import numpy as np

"""calculating dot product between two vectors that have been transformed to a higher dimension (3D)."""
x1 = np.array([3,6])
x2 = np.array([10,10])

def Transform(pt):
    x1 = pt[0]
    x2 = pt[1]
    z1 = x1**2
    z2 = np.sqrt(2) * x1 * x2
    z3 = x2**2
    return np.array([z1,z2,z3])

x1_3d = Transform(x1)
x2_3d = Transform(x2)

dot_prod = np.dot(x1_3d, x2_3d)
print(dot_prod)

"""Using the kernel function to calculate the output"""
a = np.array([3,6])
b = np.array([10,10])

kernel = a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2

print(kernel)
