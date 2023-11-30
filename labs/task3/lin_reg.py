import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

x_train = torch.rand(100)
x_train = x_train * 20 - 10

y_train = torch.pow(2, x_train) * torch.sin(torch.pow(2, (-x_train)))

noise = torch.randn(y_train.size()) / 5

y_train = y_train + noise

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.pow(2, x_validation.data) * torch.sin(torch.pow(2, (-x_validation.data)))

x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)


class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)
        # self.act2 = torch.nn.Sigmoid()
        # self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        # x = self.act2(x)
        # x = self.fc3(x)
        return x


sine_net = SineNet(25)


def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('_25_1_sq.png')
    plt.show()


# predict(sine_net, x_validation, y_validation)

optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)


def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


for epoch_index in range(2000):
    optimizer.zero_grad()

    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)

    loss_val.backward()

    optimizer.step()

predict(sine_net, x_validation, y_validation)
