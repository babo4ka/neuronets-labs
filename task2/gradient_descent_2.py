import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
print(w)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

w.to(device)

alpha = 0.001


def function(x):
    return torch.log(torch.log(x + 7)).prod()


def step(f, x):
    func_res = f(x)
    func_res.backward()
    x.data -= alpha * x.grad
    x.grad.zero_()


variables = list()
functions = list()

for t in range(500):
    variables.append(w.data.cpu().numpy().copy())
    functions.append(function(w).data.cpu().numpy().copy())
    step(function, w)

print(w)
print(variables[50])
print(functions[50])


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(x, y, functions, rstride=4, cstride=4, cmap=cm.jet)
