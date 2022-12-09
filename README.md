# neural_labs

install libs

```
pip install -r requirements.txt
```

to download mnist/cifar10 run it in any notebook

```
from torchvision import datasets
ds = datasets.MNIST(root='../data', download=True)
ds = datasets.CIFAR10(root='../data', download=True)
```
