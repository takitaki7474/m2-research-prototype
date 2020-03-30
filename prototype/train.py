import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, training, optimizers, serializers
from chainer.training import extensions

def train(args, net, train, test):
    net = L.Classifier(net)
    optimizer = optimizers.Adam(alpha=args.alpha).setup(net)

    train_iter = iterators.SerialIterator(train, args.batch_size)
    test_iter = iterators.SerialIterator(test, args.batch_size ,repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu_id)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out="./result/" + args.model_name)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, net, device=args.gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.run()
