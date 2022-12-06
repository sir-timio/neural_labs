import numpy as np


class Tanh():
    def forward(self, x):
        self.old_y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return self.old_y
    
    def backward(self, grad):
        return (1 - self.old_y**2) * grad
    

class Sigmoid():
    def forward(self, x):
        self.old_y = np.exp(x) / (1. + np.exp(x))
        return self.old_y

    def backward(self, grad):
        return self.old_y * (1. - self.old_y) * grad

class ReLU():
    def forward(self, x):
        self.old_x = np.copy(x)
        return np.clip(x,0,None)

    def backward(self, grad):
        return np.where(self.old_x>0,grad,0)
    

# use carefully
class Softmax():
    def forward(self,x):
        self.old_y = np.exp(x) / np.exp(x).sum(axis=1)[:,None]
        return self.old_y

    def backward(self,grad):
        return self.old_y * (grad -(grad * self.old_y).sum(axis=1)[:,None])