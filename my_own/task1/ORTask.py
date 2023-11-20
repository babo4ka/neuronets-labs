import matplotlib.pyplot as plt
import numpy as np

class Neuron_OR():
    def __init__(self):
        self.w1, self.w2 = 1, 1
        self.b = -0.5
        self.dots0 = list()
        self.dots1 = list()

    def operate_or(self, input_data):
        x = input_data[0]
        y = input_data[1]
        print(x, 'OR', y, "=", end=" ")
        result = x * self.w1 + y * self.w2 + self.b
        if result < 0:
            self.dots0.append([x, y])
            print(0)
            return 0
        else:
            self.dots1.append([x, y])
            print(1)
            return 1

    def get_dots(self):
        return self.dots0, self.dots1

    def clear_dots(self):
        self.dots0 = list()
        self.dots1 = list()

    def print_dots(self):
        start, end = [0, 0.5], [0.5, 0]

        xs0 = list()
        ys0 = list()
        for el in self.dots0:
            xs0.append(el[0])
            ys0.append(el[1])

        xs1 = list()
        ys1 = list()
        for el in self.dots1:
            xs1.append(el[0])
            ys1.append(el[1])

        plt.scatter(xs0, ys0, c='red')
        plt.scatter(xs1, ys1, c='green')

        plt.plot(start, end)
        plt.show()

if __name__  == "__main__":
    neuron = Neuron_OR()

    n = int(input("количество чисел: "))

    x1 = np.random.random(n)
    print(x1)
    x2 = x1 + [np.random.randint(10)/10 for i in range(n)]
    input_data = [x1, x2]

    for i in range(n):
        data = np.array([input_data[0][i], input_data[1][i]])
        neuron.operate_or(data)

    neuron.print_dots()

