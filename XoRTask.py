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

    def learn(self, data_train_in, data_train_out, neuron_not, neuron_and):
        i = 0
        for x1, x2 in data_train_in:
            temp = neuron_and.calc({'x': neuron_not.calc(x1).item(), 'y': x2}).item() * self.weights[
                0] + neuron_and.calc({'x': x1, 'y': neuron_not.calc(x2).item()}).item() * self.weights[1] + self.b * \
                   self.weights[2]
            temp = 1 / (1 + torch.exp(torch.tensor(-temp)))
            error = data_train_out[i] - temp
            i += 1
            self.weights[0] += error * x1
            self.weights[1] += error * x2
            self.weights[2] += error * self.b

    def calc(self, input_data, neuron_not, neuron_and):
        out = (((neuron_and.calc({'x': neuron_not.calc(input_data['x']).item(), 'y': input_data['y']}).item()) *
                self.weights[0] +
                neuron_and.calc({'x': input_data['x'], 'y': neuron_not.calc(input_data['y']).item()}).item() *
                self.weights[1]) +
               self.b * self.weights[2])
        return 1 / (1 + torch.exp(torch.tensor(-out)))


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

    print("НЕ", 1, "=", round(neuron_not.calc(1).item()))
    print("НЕ", 0, "=", round(neuron_not.calc(0).item()))

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
        print(data["x"], "И", data["y"], "=", round(neuron_and.calc(data).item()))

    data_train_xor = [
        {"in": [0, 0], "out": [0]},
        {"in": [0, 1], "out": [1]},
        {"in": [1, 0], "out": [1]},
        {"in": [1, 1], "out": [0]}
    ]

    train_xor_x = list(map(lambda item: item["in"], data_train_and))
    train_xor_y = list(map(lambda item: item["out"], data_train_and))

    train_xor_x = torch.tensor(train_xor_x).to(torch.float32).to(tensor_device)
    train_xor_y = torch.tensor(train_xor_y).to(torch.float32).to(tensor_device)

    neuron_xor = Neuron_XOR()

    for i in range(5000):
        neuron_xor.learn(train_xor_x, train_xor_y, neuron_not, neuron_and)

    input_data_xor = [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
        {"x": 1, "y": 1}
    ]

    for data in input_data_xor:
        print(data['x'], "XOR", data['y'], "=", neuron_xor.calc(data, neuron_not, neuron_and).item())
