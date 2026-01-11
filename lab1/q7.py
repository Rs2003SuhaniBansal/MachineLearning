theta = [2,3,3]

X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]

X_theta = []
for i in range(len(X)):
    summ = 0
    for j in range(len(theta)):
        summ += X[i][j] * theta[j]
    X_theta.append(summ)

print(X_theta)
