import torch
import matplotlib.pyplot as plt
import numpy as np

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
print(w)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

w.to(device)

alpha = 0.01

opt = torch.optim.SGD([w], lr=alpha)


def function(x):
    return torch.log(torch.log(x + 7)).prod()


def step(f, x):
    func_res = f(x)
    func_res.backward()
    opt.step()
    opt.zero_grad()


variables = list()
functions = list()

for t in range(500):
    variables.append(w.data.cpu().numpy().copy())
    functions.append(function(w).data.cpu().numpy().copy())
    step(function, w)

print(w)
print(variables[0:10])
print(functions[0:10])

def show_contours(objective,
                  x_lims=[-10.0, 10.0],
                  y_lims=[-10.0, 10.0],
                  x_ticks=100,
                  y_ticks=100):
    x_step = (x_lims[1] - x_lims[0]) / x_ticks
    y_step = (y_lims[1] - y_lims[0]) / y_ticks
    X, Y = np.mgrid[x_lims[0]:x_lims[1]:x_step, y_lims[0]:y_lims[1]:y_step]
    res = []
    for x_index in range(X.shape[0]):
        res.append([])
        for y_index in range(X.shape[1]):
            x_val = X[x_index, y_index]
            y_val = Y[x_index, y_index]
            res[-1].append(objective(torch.tensor(np.array([[x_val, y_val]]).T)))
    res = np.array(res)
    plt.figure(figsize=(7, 7))
    plt.contour(X, Y, res, 100)
    plt.xlabel('x1')
    plt.ylabel('x2')


show_contours(function)
plt.scatter(np.array(variables)[:, 0], np.array(variables)[:, 1], s=10, c='r')
plt.show()


plt.figure(figsize=(7, 7))
plt.plot(functions)
plt.xlabel('step')
plt.ylabel('function value')
plt.show()