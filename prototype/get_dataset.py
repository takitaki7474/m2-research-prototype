import numpy as np
import chainer

def cifar10():
    (train, test) = chainer.datasets.get_cifar10()
    print(len(train))
