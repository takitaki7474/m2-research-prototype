import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, training, optimizers, serializers
from chainer.training import extensions

def train(args, net, train, test):
    net = L.Classifier(net)
    optimizer = optimizers.Adam(alpha=alpha).setup(net)

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size ,repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out="logs")
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.run()
