A = [[1, 2, 3],
     [4, 5, 6]]

# A transpose
At = []
for i in range(len(A[0])):
    b = []
    for j in range(len(A)):
        b.append(A[j][i])
    At.append(b)

#A transpose A. matrix multiplication
result = []
for i in range(len(At)):
    row = []
    for j in range(len(A[0])):
        s = 0
        for k in range(len(A)):
            s += At[i][k] * A[k][j]
        row.append(s)
    result.append(row)

print(result)
