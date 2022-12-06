import numpy as np

class MSE():
    def forward(self, x, y):
        self.old_x = x
        self.old_y = y
        return (1/2 * np.square(x-y)).mean()
    
    def backward(self):
        return (self.old_x - self.old_y).mean(axis=0)
    

class CrossEntropy():
    def forward(self,x,y):
        self.old_x = x.clip(min=1e-10,max=None)
        self.old_y = y
        return (np.where(y==1,-np.log(self.old_x), 0)).mean()

    def backward(self):
        return np.where(self.old_y==1,-1/self.old_x, 0).mean(axis=0)