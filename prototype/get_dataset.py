import numpy as np
import chainer

def load_cifar10():
    (train, test) = chainer.datasets.get_cifar10()
    print(len(train))
