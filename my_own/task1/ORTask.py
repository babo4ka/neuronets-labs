import matplotlib.pyplot as plt
import numpy as np

class Neuron_OR():
    def __init__(self):
        self.w1, self.w2 = 1, 1
        self.b = -0.5
        self.dots0 = list()
        self.dots1 = list()
        for i in range(2):
            self.dots0.append(list())
            self.dots1.append(list())

    def operate_or(self, input_data):
        x = input_data[0]
        y = input_data[1]
        print(x, 'OR', y, "=", end=" ")
        result = x * self.w1 + y * self.w2 + self.b
        if result < 0:
            self.dots0[0].append(x)
            self.dots0[1].append(y)
            print(0)
        else:
            self.dots1[0].append(x)
            self.dots1[1].append(y)
            print(1)

    def print_dots(self):
        start, end = [0, 0.5], [0.5, 0]

        plt.scatter(self.dots0[0][:], self.dots0[1][:], c='red')
        plt.scatter(self.dots1[0][:], self.dots1[1][:], c='green')
        plt.plot(start, end)
        plt.show()

if __name__  == "__main__":
    neuron = Neuron_OR()

    n = int(input("количество чисел: "))

    x1 = np.random.random(n)
    x2 = x1 + [np.random.randint(10)/10 for i in range(n)]
    input_data = [x1, x2]

    for i in range(n):
        data = np.array([input_data[0][i], input_data[1][i]])
        neuron.operate_or(data)

    neuron.print_dots()