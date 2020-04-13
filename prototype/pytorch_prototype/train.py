import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os

"""
def train(args, net, train, test):
    net = L.Classifier(net)
    optimizer = optimizers.Adam(alpha=args.alpha).setup(net)

    train_iter = iterators.SerialIterator(train, args.batch_size)
    test_iter = iterators.SerialIterator(test, args.batch_size ,repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu_id)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out="./result/" + args.model_name)
    trainer.extend(extensions.LogReport(trigger=(1,'epoch')))
    trainer.extend(extensions.Evaluator(test_iter, net, device=args.gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', marker='.', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', marker='.', file_name='accuracy.png'))
    trainer.run()

    serializers.save_npz(os.path.join("./learned_model/", args.model_name) + ".model", net)
"""

def train(args, net, train, test):

    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    trainiter = iter(trainloader)
    testiter = iter(testloader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.alpha, momentum=0.9)

    for epoch in range(args.epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
