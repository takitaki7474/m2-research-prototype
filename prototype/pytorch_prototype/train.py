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

        out_result.append(sum_loss*args.batch_size/len(testloader.dataset))
        out_result.append(float(sum_correct/sum_data_n))
        test_loss_value.append(sum_loss*args.batch_size/len(testloader.dataset))
        test_acc_value.append(float(sum_correct/sum_data_n))

        value_log(log={
            "epoch": epoch+1,
            "train loss": out_result[0],
            "train accuracy": out_result[1],
            "test loss": out_result[2],
            "test accuracy": out_result[3]
        }) # logの追加
        print("{0:<13}{1:<13.5f}{2:<13.5f}{3:<13.5f}{4:<13.5f}".format(epoch+1, out_result[0], out_result[1], out_result[2], out_result[3]))
        out_result = []

    value_log(save_flag=1) # logの保存
    torch.save(net.state_dict(), model_save_path)

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

# def train_or_test(loader, optimizer, net, criterion):
