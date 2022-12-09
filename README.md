# neural_labs

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
