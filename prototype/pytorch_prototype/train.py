import log
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

def lr_scheduling(epoch):
    if epoch < 20:
        return 1
    elif epoch < 40:
        return 0.1**1
    elif epoch < 60:
        return 0.1**2
    else:
        return 0.1**3

def train(args, net, train, test):

    train_loss_value = []
    train_acc_value = []
    test_loss_value = []
    test_acc_value = []

    value_log = log.save_log_func(args)

    model_save_path = os.path.join("./learned_model", args.model_name + ".pth")
    result_save_path = os.path.join("./result",args.model_name)

    if not os.path.isdir(result_save_path):
        os.mkdir(result_save_path)

    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.alpha, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduling)

    print("{0:<13}{1:<13}{2:<13}{3:<13}{4:<13}{5:<6}".format("epoch","train/loss","train/acc","test/loss","test/acc","lr"))

    for epoch in range(args.epoch):

        result = [] # 出力結果
        sum_loss = 0.0  # lossの合計
        sum_correct = 0 # 正解数の合計
        sum_data_n = 0  # データ数の合計

        for (inputs, labels) in trainloader:
            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_data_n += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

        for param in optimizer.param_groups:
            current_lr = param["lr"]

        scheduler.step()
        result.append(sum_loss*args.batch_size/len(trainloader.dataset))
        result.append(float(sum_correct/sum_data_n))
        train_loss_value.append(sum_loss*args.batch_size/len(trainloader.dataset))
        train_acc_value.append(float(sum_correct/sum_data_n))

        sum_loss = 0.0
        sum_correct = 0
        sum_data_n = 0

        for (inputs, labels) in testloader:
            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_data_n += labels.size(0)
            sum_correct += (predicted == labels).sum().item()

        result.append(sum_loss*args.batch_size/len(testloader.dataset))
        result.append(float(sum_correct/sum_data_n))
        result.append(current_lr)
        test_loss_value.append(sum_loss*args.batch_size/len(testloader.dataset))
        test_acc_value.append(float(sum_correct/sum_data_n))

        value_log(log={
            "epoch": epoch+1,
            "train loss": result[0],
            "train accuracy": result[1],
            "test loss": result[2],
            "test accuracy": result[3],
            "learning rate": result[4]
        }) # logの追加
        print("{0:<13}{1:<13.5f}{2:<13.5f}{3:<13.5f}{4:<13.5f}{5:<6.6f}".format(epoch+1, result[0], result[1], result[2], result[3], result[4]))
        result = []

    value_log(save_flag=1) # logの保存
    torch.save(net.state_dict(), model_save_path)
    plot_value(args, result_save_path, train_loss_value, test_loss_value, train_acc_value, test_acc_value)


def plot_value(args, result_save_path, train_loss_value, test_loss_value, train_acc_value, test_acc_value):

    plt.figure(figsize=(7,5))
    plt.plot(range(args.epoch), train_loss_value)
    plt.plot(range(args.epoch), test_loss_value, c='#ed7700')
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'test loss'])
    plt.grid()
    plt.savefig(os.path.join(result_save_path, "loss.png"))
    plt.clf()

    plt.plot(range(args.epoch), train_acc_value)
    plt.plot(range(args.epoch), test_acc_value, c='#ed7700')
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train acc', 'test acc'])
    plt.grid()
    plt.savefig(os.path.join(result_save_path, "accuracy.png"))
    plt.clf()
