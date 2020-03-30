import chainer
import chainer.links as L
import chainer.functions as F

class Cifar10Model(chainer.Chain):

    def __init__(self, n_out=3):
        super(Cifar10Model,self).__init__()
        with self.init_scope():
            conv1 = F.Convolution2D(3, 32, 3, pad=1),
            conv2 = F.Convolution2D(32, 32, 3, pad=1),
            conv3 = F.Convolution2D(32, 32, 3, pad=1),
            conv4 = F.Convolution2D(32, 32, 3, pad=1),
            conv5 = F.Convolution2D(32, 32, 3, pad=1),
            conv6 = F.Convolution2D(32, 32, 3, pad=1),
            l1 = L.Linear(512, 512),
            l2 = L.Linear(512, n_out)

    def __call__(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), train=train)
        return self.l2(h)
