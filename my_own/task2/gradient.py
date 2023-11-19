import numpy as np
import torch

class Gradient():
    def __init__(self):
        self.w = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.alpha = 0.001

    def set_w(self, w):
        self.w = torch.tensor(w, requires_grad=True, dtype=torch.float32)
        self.w.to(self.device)

    def function(self, x):
        return torch.log(torch.log(x + 7)).prod()

    def step(self, f, x):
        func_res = f(x)
        func_res.backward()
        x.data -= self.alpha * x.grad
        x.grad.zero_()

    def get_grad_in_dot(self):
        result = self.function(self.w)
        result.backward()
        return self.w.grad

    def get_grad_after_descent(self, steps):
        for i in range(steps):
            self.step(self.function, self.w)
        return self.w.data

