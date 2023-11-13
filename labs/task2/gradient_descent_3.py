import torch

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
print(w)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

w.to(device)

alpha = 0.001

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
