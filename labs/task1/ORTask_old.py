import matplotlib.pyplot as plt
import numpy
import random
import pandas as pd

class Neuron_OR():
    def __init__(self):
        self.weights = list()
        self.b = 1
        self.temps = list()
        for i in range(3):
            self.weights.append(random.random())

    def learn(self, x1, x2, out):
        temp = x1 * self.weights[0] + x2 * self.weights[1] + self.b * self.weights[2]
        temp = 1 / (1 + numpy.exp(-temp))
        self.temps.append(temp)
        error = out - temp
        self.weights[0] += error * x1
        self.weights[1] += error * x2
        self.weights[2] += error * self.b

    def calc(self, input_data):
        xs = list()
        ys = list()
        outs = list()
        for x, y in input_data:
            out = x * self.weights[0] + y * self.weights[1] + self.b * self.weights[2]
            out = 1 / (1 + numpy.exp(-out))
            xs.append(x)
            ys.append(y)
            outs.append(out)
        return pd.DataFrame({
            "x": xs,
            "y": ys,
            "out": outs
        })


if __name__ == '__main__':
    neuron = Neuron_OR()

    for j in range(5000):
        neuron.learn(1, 1, 1)
        neuron.learn(1, 0, 1)
        neuron.learn(0, 1, 1)
        neuron.learn(0, 0, 0)

    result = neuron.calc([[0, 0], [0, 1], [1, 0], [1, 1]])

    for index, row in result.iterrows():
        print(row["x"].astype('int32'), "ИЛИ", row["y"].astype('int32'), "=", row["out"])

    plt.plot(neuron.temps)
    plt.show()