# NN from scratch

lab1 - MLP

lab2 - optimization algorithms (Momentum, Nestrov, AdaGrad, AdaDelta, RMSprop, Adam)

lab3 - l1,l2 regularization, dropout

lab4 - K-means and radial bases NN

lab5 - AlexNet, LeNet, ResNet, ResNeXt with torch

lab6 - lstm forecast with torch

install libs (python 3.7.10)

```
pip install -r requirements.txt
```
if case of error remove lib version and run

```
pip install [PACKAGE] -U
```


to download mnist/cifar10 run it in any notebook

```
from torchvision import datasets
ds = datasets.MNIST(root='../data', download=True)
ds = datasets.CIFAR10(root='../data', download=True)
```
