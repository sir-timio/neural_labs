import numpy as np
from copy import deepcopy


class SGD():
    def __init__(self, lr=0.01, l1=0, l2=0):
        self.lr = lr
        self.l1 = l1
        self.l2 = l2

    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]

    def step(self):
        for layer in self.layers:
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)
            
            layer.w -= self.lr * layer.grad_w
            layer.b -= self.lr * layer.grad_b
        

class Momentum():
    def __init__(self, lr=0.01, momentum=0.6, l1=0, l2=0):
        self.m = momentum
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]
        
        self.velocity_w = []
        self.velocity_b = []
        for l in self.layers:
            self.velocity_w.append(np.zeros_like(l.w))
            self.velocity_b.append(np.zeros_like(l.b))
   
    def step(self):
        for i, layer in enumerate(self.layers):
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)
            
            self.velocity_w[i] = self.m * self.velocity_w[i] + self.lr * layer.grad_w
            self.velocity_b[i] = self.m * self.velocity_b[i] + self.lr * layer.grad_b
            layer.w -= self.velocity_w[i]
            layer.b -= self.velocity_b[i]

class Nesterov():
    def __init__(self, lr=0.01, momentum=0.6, l1=0, l2=0):
        self.m = momentum
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]
        
        self.velocity_w = []
        self.velocity_b = []
        for l in self.layers:
            self.velocity_w.append(np.zeros_like(l.w))
            self.velocity_b.append(np.zeros_like(l.b))
        
    def step(self):
        model_ahead =  deepcopy(self.model)
        ahead_layers = [l  for l in model_ahead.layers if type(l) == Linear]
        
        for i, layer in enumerate(ahead_layers):
            layer.w -= self.m * self.velocity_w[i] 
            layer.b -= self.m * self.velocity_b[i]
        
        
        model_ahead.loss(self.model.cur_x, self.model.cur_y)
        model_ahead.backward()
        
        ahead_layers = [l  for l in model_ahead.layers if type(l) == Linear]
        
        for i, (ahead_layer, layer) in enumerate(zip(ahead_layers, self.layers)):
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)
            
            self.velocity_w[i] = self.m * self.velocity_w[i] + self.lr * ahead_layer.grad_w
            self.velocity_b[i] = self.m * self.velocity_b[i] + self.lr * ahead_layer.grad_b
            
            layer.w -= self.velocity_w[i]
            layer.b -= self.velocity_b[i]

class AdaGrad():
    def __init__(self, lr=1, l1=0, l2=0):
        self.lr = lr
        self.eps = 1e-10
        self.l1 = l1
        self.l2 = l2
        
    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]
        
        self.N_w = []
        self.N_b = []
        for l in self.layers:
            self.N_w.append(np.zeros_like(l.w))
            self.N_b.append(np.zeros_like(l.b))
   
    def step(self):
        for i, layer in enumerate(self.layers):
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)
            
            self.N_w[i] += layer.grad_w ** 2 
            self.N_b[i] += layer.grad_b ** 2
            
            layer.w -= self.lr * layer.grad_w / (np.sqrt(self.N_w[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.N_b[i]) + self.eps)

class RMSprop():
    def __init__(self, lr=0.01, decay=0.9, l1=0, l2=0):
        self.lr = lr
        self.decay = decay
        self.eps = 1e-10
        self.l1 = l1
        self.l2 = l2
        
    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]
        
        self.N_w = []
        self.N_b = []
        for l in self.layers:
            self.N_w.append(np.zeros_like(l.w))
            self.N_b.append(np.zeros_like(l.b))
   
    def step(self):
        for i, layer in enumerate(self.layers):
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)
            
            self.N_w[i] = self.decay * self.N_w[i] + (1-self.decay) * layer.grad_w ** 2
            self.N_b[i] = self.decay * self.N_b[i] + (1-self.decay) * layer.grad_b ** 2
            
            layer.w -= self.lr * layer.grad_w / (np.sqrt(self.N_w[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.N_b[i]) + self.eps)

class AdaDelta():
    def __init__(self, lr=0.01, decay=0.9, l1=0, l2=0):
        self.lr = lr
        self.decay = decay
        self.l1 = l1
        self.l2 = l2
        self.eps = 1e-10
    
    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]
        
        self.N_w = []
        self.N_b = []
        
        self.P_w = []
        self.P_b = []
        for l in self.layers:
            self.N_w.append(np.zeros_like(l.w))
            self.N_b.append(np.zeros_like(l.b))
            
            self.P_w.append(np.zeros_like(l.w))
            self.P_b.append(np.zeros_like(l.b))
            

    def step(self):
        for i, layer in enumerate(self.layers):
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)
            
            self.N_w[i] = self.decay * self.N_w[i] + (1-self.decay)*layer.grad_w ** 2 
            self.N_b[i] = self.decay * self.N_b[i] + (1-self.decay)*layer.grad_b ** 2

            d_w = layer.grad_w * np.sqrt(self.P_w[i] + self.eps) / np.sqrt(self.N_w[i] + self.eps)
            d_b = layer.grad_b * np.sqrt(self.P_b[i] + self.eps) / np.sqrt(self.N_b[i] + self.eps)

            self.P_w[i] = self.decay * self.P_w[i] + (1 - self.decay) * d_w ** 2
            self.P_b[i] = self.decay * self.P_b[i] + (1 - self.decay) * d_b ** 2
            layer.w -= self.lr * d_w
            layer.b -= self.lr * d_b
            
class Adam():
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, l1=0, l2=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-10
        self.l1 = l1
        self.l2 = l2
    
    def init_params(self, model):
        self.model = model
        self.layers = [l  for l in model.layers if hasattr(l, 'requires_grad') and l.requires_grad]
                
        self.M_w = []
        self.M_b = []
        
        self.N_w = []
        self.N_b = []
        for l in self.layers:              
            self.M_w.append(np.zeros_like(l.w))
            self.M_b.append(np.zeros_like(l.b))
            
            self.N_w.append(np.zeros_like(l.w))
            self.N_b.append(np.zeros_like(l.b))
            
   
    def step(self):
        t = self.model.cur_epoch + 1
        for i, layer in enumerate(self.layers):
            layer.grad_w += self.l1 * layer.w + self.l2 * np.sign(layer.w)

            self.M_w[i] = self.beta1 * self.M_w[i] + (1 - self.beta1) * layer.grad_w
            self.M_b[i] = self.beta1 * self.M_b[i] + (1 - self.beta1) * layer.grad_b


            self.N_w[i] = self.beta2 * self.N_w[i] + (1 - self.beta2) * layer.grad_w ** 2
            self.N_b[i] = self.beta2 * self.N_b[i] + (1 - self.beta2) * layer.grad_b ** 2
            
            m_w_hat = self.M_w[i] / (1 - self.beta1 ** t)
            m_b_hat = self.M_b[i] / (1 - self.beta1 ** t)
            
            n_w_hat = self.N_w[i] / (1 - self.beta2 ** t)
            n_b_hat = self.N_b[i] / (1 - self.beta2 ** t)


            layer.w -= self.lr * m_w_hat / (np.sqrt(n_w_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(n_b_hat) + self.eps)