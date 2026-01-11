import matplotlib.pyplot as plt

start = -100
stop = 100
num = 100

step  = (stop - start)/(num - 1)

x1 = []
for i in range(num):
    x1.append(start+i*step)

y = []
for n in x1:
    y.append(2 * n + 3)
    
plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = 2x1 + 3")
plt.show()


