import torch
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class Regression:
    def __init__(self):
        self.x_train = torch.rand(100)
        self.x_train = self.x_train * 20 - 10

        self.y_train = torch.pow((torch.cos(self.x_train + 3)), 2) * torch.pow(2, self.x_train - 3)

        self.noise = torch.randn(self.y_train.size()) / 5

        self.y_train = self.y_train + self.noise

        self.x_train.unsqueeze_(1)
        self.y_train.unsqueeze_(1)

        self.net = Net(50)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

    def loss(self, pred, target):
        squares = (pred - target) ** 2
        return squares.mean()

    def learn(self):
        for epoch_index in range(2000):
            self.optimizer.zero_grad()

            y_pred = self.net.forward(self.x_train)
            loss_val = self.loss(y_pred, self.y_train)

            loss_val.backward()

            self.optimizer.step()

    def predict(self, x):
        return self.net.forward(x)



