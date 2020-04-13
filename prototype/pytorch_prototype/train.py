import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def train(args, net, train, test):

    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.alpha, momentum=0.9)

    train_loss_value = []
    train_acc_value = []
    test_loss_value = []
    test_acc_value = []

    print("{0:<13}{1:<13}{2:<13}{3:<13}{4:<13}".format("epoch","train/loss","train/acc","test/loss","test/acc"))

    for epoch in range(args.epoch):

        out_result = [] # 出力結果
        sum_loss = 0.0  # lossの合計
        sum_correct = 0 # 正解数の合計
        sum_data_n = 0  # データ数の合計

        for (inputs, labels) in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_data_n += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

        out_result.append(sum_loss*args.batch_size/len(trainloader.dataset))
        out_result.append(float(sum_correct/sum_data_n))
        train_loss_value.append(sum_loss*args.batch_size/len(trainloader.dataset))
        train_acc_value.append(float(sum_correct/sum_data_n))

        sum_loss = 0.0
        sum_correct = 0
        sum_data_n = 0

        for (inputs, labels) in testloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_data_n += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

        out_result.append(sum_loss*args.batch_size/len(testloader.dataset))
        out_result.append(float(sum_correct/sum_data_n))
        test_loss_value.append(sum_loss*args.batch_size/len(testloader.dataset))
        test_acc_value.append(float(sum_correct/sum_data_n))

        print("{0:<13}{1:<13.5f}{2:<13.5f}{3:<13.5f}{4:<13.5f}".format(epoch, out_result[0], out_result[1], out_result[2], out_result[3]))
        out_result = []

"""
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

            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
"""
    #print('Finished Training')
