import random
import torch

tensor_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class Neuron_NOT():
    def __init__(self):
        self.weights = list()
        self.b = 1
        for i in range(2):
            self.weights.append(random.random())

    def learn(self, data_train_x, data_train_out):
        for x, out in data_train_x, data_train_out:
            temp = x * self.weights[0] + self.b * self.weights[1]
            temp = 1 / (1 + torch.exp(torch.Tensor(-temp)))
            error = out - temp
            self.weights[0] += error * x
            self.weights[1] += error * self.b

    def calc(self, x):
        out = x * self.weights[0] + self.b * self.weights[1]
        return 1 / (1 + torch.exp(torch.Tensor(-out)))


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
            temp = 1 / (1 + torch.exp(torch.Tensor(-temp)))
            error = data_train_out[i] - temp
            i += 1
            self.weights[0] += error * x1
            self.weights[1] += error * x2
            self.weights[2] += error * self.b

    def calc(self, input_data):
        out = (input_data['x'] * self.weights[0]) * (input_data['y'] * self.weights[1]) + self.b * self.weights[2]
        return 1 / (1 + torch.exp(torch.Tensor(-out)))


class Neuron_OR():
    def __init__(self):
        self.weights = list()
        self.b = 1
        for i in range(3):
            self.weights.append(random.random())

    def learn(self, data_train_in, data_train_out):
        i = 0
        for x1, x2 in data_train_in:
            temp = x1 * self.weights[0] + x2 * self.weights[1] + self.b * self.weights[2]
            temp = 1 / (1 + torch.exp(torch.Tensor(-temp)))
            error = data_train_out[i] - temp
            i += 1
            self.weights[0] += error * x1
            self.weights[1] += error * x2
            self.weights[2] += error * self.b

    def calc(self, input_data):
        out = input_data['x'] * self.weights[0] + input_data['y'] * self.weights[1] + self.b * self.weights[2]
        return 1 / (1 + torch.exp(torch.Tensor(-out)))


if __name__ == '__main__':
    # нейрон НЕ

    # данные для обучения
    data_train_not = [
        {"in": [1], "out": [0]},
        {"in": [0], "out": [1]}
    ]

    tensor_train_x_not = list(map(lambda item: item["in"], data_train_not))
    tensor_train_y_not = list(map(lambda item: item["out"], data_train_not))

    tensor_train_x_not = torch.tensor(tensor_train_x_not).to(torch.float32).to(tensor_device)
    tensor_train_y_not = torch.tensor(tensor_train_y_not).to(torch.float32).to(tensor_device)

    neuron_not = Neuron_NOT()

    # обучение
    for i in range(5000):
        neuron_not.learn(tensor_train_x_not, tensor_train_y_not)

    # результат
    print("НЕ", 1, "=", round(neuron_not.calc(1).item()))
    print("НЕ", 0, "=", round(neuron_not.calc(0).item()))

    print("==============================================")

    # входные значения для нейрона
    input_data = [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},
        {"x": 1, "y": 0},
        {"x": 1, "y": 1}
    ]

    # нейрон И

    # данные для обучения
    data_train_and = [
        {"in": [0, 0], "out": [0]},
        {"in": [0, 1], "out": [0]},
        {"in": [1, 0], "out": [0]},
        {"in": [1, 1], "out": [1]}
    ]

    tensor_train_x_and = list(map(lambda item: item["in"], data_train_and))
    tensor_train_y_and = list(map(lambda item: item["out"], data_train_and))

    tensor_train_x_and = torch.tensor(tensor_train_x_and).to(torch.float32).to(tensor_device)
    tensor_train_y_and = torch.tensor(tensor_train_y_and).to(torch.float32).to(tensor_device)

    neuron_and = Neuron_AND()

    # обучение
    for i in range(5000):
        neuron_and.learn(tensor_train_x_and, tensor_train_y_and)

    # результат
    for data in input_data:
        print(data["x"], "И", data["y"], "=", round(neuron_and.calc(data).item()))

    print("==============================================")

    # нейрон ИЛИ

    # данные для обучения
    data_train_or = [
        {"in": [0, 0], "out": [0]},
        {"in": [0, 1], "out": [1]},
        {"in": [1, 0], "out": [1]},
        {"in": [1, 1], "out": [1]}
    ]

    tensor_train_x_or = list(map(lambda item: item["in"], data_train_or))
    tensor_train_y_or = list(map(lambda item: item["out"], data_train_or))

    tensor_train_x_or = torch.tensor(tensor_train_x_or).to(torch.float32).to(tensor_device)
    tensor_train_y_or = torch.tensor(tensor_train_y_or).to(torch.float32).to(tensor_device)

    neuron_or = Neuron_OR()

    # обучение
    for i in range(5000):
        neuron_or.learn(tensor_train_x_or, tensor_train_y_or)

    # результат
    for data in input_data:
        print(data["x"], "ИЛИ", data["y"], "=", round(neuron_or.calc(data).item()))

    print("==============================================")

    #результат операции XOR
    for data in input_data:
        print(data["x"], "XOR", data["y"], "=", round(neuron_or.calc(
            {'x': neuron_and.calc({'x': neuron_not.calc(data['x']).item(), 'y': data['y']}).item(),
             'y': neuron_and.calc({'x': data['x'], 'y': neuron_not.calc(data['y']).item()}).item()}).item()))
