import numpy
import torch
import random
import pandas as pd
from torch import nn

tensor_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#
# data_train = [
#     {"in": [0, 0], "out": [0]},
#     {"in": [0, 1], "out": [1]},
#     {"in": [1, 0], "out": [1]},
#     {"in": [1, 1], "out": [0]},
# ]
#
# tensor_train_x = list(map(lambda item: item["in"], data_train))
# tensor_train_y = list(map(lambda item: item["out"], data_train))
#
# tensor_train_x = torch.tensor(tensor_train_x).to(torch.float32).to(tensor_device)
# tensor_train_y = torch.tensor(tensor_train_y).to(torch.float32).to(tensor_device)
#
# print("ввод: ")
# print(tensor_train_x)
# print("shape: ", tensor_train_x.shape)
# print()
# print("вывод: ")
# print(tensor_train_y)
# print("shape: ", tensor_train_y.shape)


class Neuron_NOT():
    def __init__(self):
        self.weights = list()
        self.b = 1
        for i in range(2):
            self.weights.append(random.random())

    def learn(self, data_train_x, data_train_out):
        for x, out in data_train_x, data_train_out:
            temp = x * self.weights[0] + self.b * self.weights[1]
            temp = 1 / (1 + torch.exp(torch.tensor(-temp)))
            error = out - temp
            self.weights[0] += error * x
            self.weights[1] += error * self.b

    def calc(self, x):
        out = x * self.weights[0] + self.b * self.weights[1]
        return 1 / (1 + torch.exp(torch.tensor(-out)))


class Neuron_AND():
    def __init__(self):
        self.weights = list()
        self.b = 1
        for i in range(3):
            self.weights.append(random.random())

    def learn(self, data_train_in, data_train_out):
        i = 0
        for x1, x2 in data_train_in:
            temp = (x1 * self.weights[0]) * (x2 * self.weights[1]) + self.b * self.weights[2]
            temp = 1 / (1 + torch.exp(torch.tensor(-temp)))
            error = data_train_out[i] - temp
            i += 1
            self.weights[0] += error * x1
            self.weights[1] += error * x2
            self.weights[2] += error * self.b

    def calc(self, input_data):
        out = (input_data['x'] * self.weights[0]) * (input_data['y'] * self.weights[1]) + self.b * self.weights[2]
        return 1 / (1 + torch.exp(torch.tensor(-out)))


class Neuron_XOR():
    def __init__(self):
        self.weights = list()
        self.b = 1
        for i in range(3):
            self.weights.append(random.random())

    def learn(self, x1, x2, out, neuron_not, neuron_and):
        temp = (neuron_and.calc(
            pd.DataFrame({
                "x": [neuron_not.calc(x1)],
                "y": [x2]
            })) * self.weights[0] +
                neuron_and.calc(
                    pd.DataFrame({
                        "x": [x1],
                        "y": [neuron_not.calc(x2)]
                    })) * self.weights[1] +
                self.b * self.weights[2])
        temp = 1 / (1 + numpy.exp(-temp))
        error = out - temp
        self.weights[0] += error * x1
        self.weights[1] += error * x2
        self.weights[2] += error * self.b

    def calc(self, input_data, neuron_not, neuron_and):
        out = (neuron_and.calc(
            pd.DataFrame({
                "x": [neuron_not.calc(input_data["x"].iloc[0])],
                "y": [input_data["y"].iloc[0]]
            })) * self.weights[0] +
               neuron_and.calc(
                   pd.DataFrame({
                       "x": [input_data["x"].iloc[0]],
                       "y": [neuron_not.calc(input_data["y"].iloc[0])]
                   })) * self.weights[1] +
               self.b * self.weights[2])
        return 1 / (1 + numpy.exp(-out))


if __name__ == '__main__':
    data_train_not = [
        {"in": [1], "out": [0]},
        {"in": [0], "out": [1]}
    ]

    tensor_train_x = list(map(lambda item: item["in"], data_train_not))
    tensor_train_y = list(map(lambda item: item["out"], data_train_not))

    tensor_train_x = torch.tensor(tensor_train_x).to(torch.float32).to(tensor_device)
    tensor_train_y = torch.tensor(tensor_train_y).to(torch.float32).to(tensor_device)

    neuron_not = Neuron_NOT()

    for i in range(5000):
        neuron_not.learn(tensor_train_x, tensor_train_y)

    print("НЕ", 1, "=", neuron_not.calc(1).item())
    print("НЕ", 0, "=", neuron_not.calc(0).item())

    print("==============================================")

    data_train_and = [
        {"in": [0, 0], "out": [0]},
        {"in": [0, 1], "out": [0]},
        {"in": [1, 0], "out": [0]},
        {"in": [1, 1], "out": [1]}
    ]

    train_and_x = list(map(lambda item: item["in"], data_train_and))
    train_and_y = list(map(lambda item: item["out"], data_train_and))

    train_and_x = torch.tensor(train_and_x).to(torch.float32).to(tensor_device)
    train_and_y = torch.tensor(train_and_y).to(torch.float32).to(tensor_device)


    neuron_and = Neuron_AND()

    for i in range(5000):
        neuron_and.learn(train_and_x, train_and_y)

    input_data_and = [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
        {"x": 1, "y": 1}
    ]

    for data in input_data_and:
        print(data["x"], "И", data["y"], "=", neuron_and.calc(data).item())

    # neuron_not = Neuron_NOT()
    #
    # for j in range(500):
    #     neuron_not.learn(1, 0)
    #     neuron_not.learn(0, 1)
    #
    # print("НЕ", 1, "=", neuron_not.calc(1))
    # print("НЕ", 0, "=", neuron_not.calc(0))
    #
    # print("===================================================")
    #
    # neuron_and = Neuron_AND()
    #
    # for j in range(500):
    #     neuron_and.learn(0, 0, 0)
    #     neuron_and.learn(1, 0, 0)
    #     neuron_and.learn(0, 1, 0)
    #     neuron_and.learn(1, 1, 1)
    #
    # for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    #     print(x, "И", y, "=", neuron_and.calc(pd.DataFrame({"x": [x], "y": [y]})).iloc[0])
    #
    # print("===================================================")
    #
    # neuron_xor = Neuron_XOR()
    #
    # for j in range(500):
    #     neuron_xor.learn(1, 1, 0, neuron_not, neuron_and)
    #     neuron_xor.learn(0, 1, 1, neuron_not, neuron_and)
    #     neuron_xor.learn(1, 0, 1, neuron_not, neuron_and)
    #     neuron_xor.learn(0, 0, 0, neuron_not, neuron_and)
    #
    #
    #
    # for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    #     print(x, "XOR", y, "=", neuron_xor.calc(pd.DataFrame({
    #         "x": [x], "y": [y]
    #     }), neuron_not, neuron_and).iloc[0])
