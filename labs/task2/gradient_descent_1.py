import torch

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
print(w)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

w.to(device)

function = torch.log(torch.log(w + 7)).prod()

function.backward()

print(w.grad)
