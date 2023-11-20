import numpy as np
import matplotlib.pyplot as plt


class Neuron_XOR():
    def __init__(self):
        self.w1 = [1, 1, -1.5]
        self.w2 = [1, 1, -0.5]
        self.dots0 = list()
        self.dots1 = list()

    def func_act(self, x):
        return 0 if x <= 0 else 1

    def operate_xor(self, input_data):
        x1 = input_data[0]
        x2 = input_data[1]
        x = np.array([x1, x2, 1])
        hidden_neuron = np.array([self.w1, self.w2])
        w_out_neuron = np.array([-1, 1, -0.5])

        sum = np.dot(hidden_neuron, x)
        out = [self.func_act(x) for x in sum]
        out.append(1)
        out = np.array(out)

        sum = np.dot(w_out_neuron, out)

        result = self.func_act(sum)
        if result > 0:
            self.dots1.append([x1, x2])
        else:
            self.dots0.append([x1, x2])

        return result

    def get_dots(self):
        return self.dots0, self.dots1

    def clear_dots(self):
        self.dots0 = list()
        self.dots1 = list()

    def print_dots(self):
        start1, end1 = [0, 0.5], [0.5, 0]
        start2, end2 = [0.5, 1], [1, 0.5]

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

        plt.plot(start1, end1, start2, end2)
        plt.scatter(xs0, ys0, c='red')
        plt.scatter(xs1, ys1, c='green')
        plt.show()


if __name__ == "__main__":
    neuron = Neuron_XOR()

    n = int(input("количество чисел: "))

    x1 = np.random.random(n)
    x2 = x1 + [np.random.randint(10)/10 for i in range(n)]
    input_data = [x1, x2]

    for i in range(n):
        result = neuron.operate_xor([input_data[0][i], input_data[1][i]])
        print(result)

    neuron.print_dots()