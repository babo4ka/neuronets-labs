import matplotlib.pyplot as plt

w1, w2 = 1, 1
b = -0.5

x1 = [1, 1, 0, 0]
x2 = [1, 0, 1, 0]

input = [x1, x2]

dots0 = list()
dots0.append(list())
dots0.append((list()))

dots1 = list()
dots1.append(list())
dots1.append((list()))

for i in range(4):
    x = input[0][i]
    y = input[1][i]
    print(x, 'OR', y, "=", end=" ")
    result = x*w1 + y*w2 + b
    if result < 0:
        dots0[0].append(x)
        dots0[1].append(y)
        print(0)
    else:
        dots1[0].append(x)
        dots1[1].append(y)
        print(1)

line = [-b, b]


plt.scatter(dots0[0][:], dots0[1][:], c='red')
plt.scatter(dots1[0][:], dots1[1][:], c='green')
plt.plot(line)
plt.show()
