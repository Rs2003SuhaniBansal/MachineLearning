import numpy as np

X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]

n = len(X)
m = len(X[0])

mean = [sum(row[j] for row in X) / n for j in range(m)]

cov = [[0]*m for _ in range(m)]

for i in range(m):
    for j in range(m):
        for k in range(n):
            cov[i][j] += (X[k][i] - mean[i]) * (X[k][j] - mean[j])
        cov[i][j] = cov[i][j] / (n - 1)

for row in cov:
    print(row)

# using numpy

X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])
sample_num = X.shape[0]
mean = np.mean(X, axis=0)

cov = ((X - mean).T @ (X-mean))/(sample_num -1)
print(cov)