import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets
import time

seed = 2

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train = X_train.float()
X_test = X_test.float()

# plt.imshow(X_train[0, :, :])
# plt.show()
print(y_train[0])

X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])


class MNISTNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


mnist_net = MNISTNet(100)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
mnist_net = mnist_net.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mnist_net.parameters(), lr=1.0e-3, momentum=0.8)

batch_size = 100

test_accuracy_history = []
test_loss_history = []

train_accuracy_history = []
train_loss_history = []

X_test = X_test.to(device)
y_test = y_test.to(device)

X_train = X_train.to(device)
y_train = y_train.to(device)

start = time.time()
for epoch in range(100):
    order = np.random.permutation(len(X_train))

    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes].to(device)  # to(device)
        y_batch = y_train[batch_indexes].to(device)  # to(device)

        preds = mnist_net.forward(X_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = mnist_net.forward(X_test)
    loss_val = loss(test_preds, y_test)
    test_loss_history.append(loss_val.detach().cpu())

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    test_accuracy_history.append(accuracy.cpu())
    print("valid", accuracy)

    train_preds = mnist_net.forward(X_train)
    train_loss_history.append(loss(train_preds, y_train).detach().cpu())
    train_accuracy = (train_preds.argmax(dim=1) == y_train).float().mean()
    train_accuracy_history.append(train_accuracy.cpu())

end = time.time()

print('время:', end - start)
print(device)
plt.plot(test_loss_history, c='green', label='потери validation')
plt.plot(train_loss_history, c='red', label='потери train')
plt.legend(loc='upper left')
plt.show()
