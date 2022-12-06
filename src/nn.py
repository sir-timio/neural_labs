import numpy as np


class Linear():
    def __init__(self, n_in, n_out):
        self.w = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.b = np.zeros(n_out)
        self.requires_grad = True
        
    def forward(self, x):
        self.old_x = x
        return np.dot(x, self.w) + self.b

    def backward(self,grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = self.old_x.T @ grad
        return np.dot(grad, self.w.T)