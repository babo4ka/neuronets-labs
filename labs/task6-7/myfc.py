import torch.nn as nn
import torch

x = torch.tensor([10., 20.])

fc = nn.Linear(2, 3)

w = torch.tensor([[11., 12,], [21., 22.], [31., 32.]])
fc.weight.data = w

b = torch.tensor([31., 32., 33.])
fc.bias.data = b

fc_out = fc(x)

fc_out_alt = x.matmul(w.T) + b

print(fc_out == fc_out_alt)


fc_out_sum = fc_out.sum()
fc_out_sum.backward()
weight_grad = fc.weight.grad
bias_grad = fc.bias.grad


w.requires_grad_(True)
b.requires_grad_(True)

our_formula = (x.matmul(w.T) + b).sum()

our_formula.backward()

print('fc_weight_grad:', weight_grad)
print('our_weight_grad:', w.grad)
print('fc_bias_grad:', bias_grad)
print('out_bias_grad:', b.grad)
