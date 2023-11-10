import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

x_train = torch.rand(100)
x_train = x_train * 20 - 10

y_train = torch.pow(2, x_train) * torch.sin(torch.pow(2, (-x_train)))

# plt.plot(x_train.numpy(), y_train.numpy(), 'o')
# plt.title("y = 2^x * sin(2^(-x))")
# plt.show()

noise = torch.randn(y_train.size()) / 5

# plt.plot(x_train.numpy(), noise.numpy(), 'o')
# plt.title('Gaussian noise')
# plt.show()


y_train = y_train + noise

# plt.plot(x_train.numpy(), y_train.numpy(), 'o')
# plt.title('noisy 2^x * sin(2^(-x))')
# plt.xlabel('x_train')
# plt.ylabel('y_train')
# plt.show()

x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.pow(2, x_validation.data) * torch.sin(torch.pow(2, (-x_validation.data)))

# plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
# plt.title('2^x * sin(2^(-x))')
# plt.xlabel('x_validation')
# plt.ylabel('y_validation')
# plt.show()


x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


sine_net = SineNet(10)


def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


predict(sine_net, x_validation, y_validation)
