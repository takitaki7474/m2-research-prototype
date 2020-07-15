import torch
import torch.nn as nn
import torch.optim as optim

def train(train, test, net, args, lr_scheduling=None):

    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.alpha, momentum=0.9)
    if lr_scheduling is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduling)

    print("{0:<13}{1:<13}{2:<13}{3:<13}{4:<13}{5:<6}".format("epoch","train/loss","train/acc","test/loss","test/acc","lr"))

    for epoch in range(args.epoch):

        # 学習モデルの訓練
        sum_loss = 0.0  # lossの合計
        sum_correct = 0 # 正解数の合計
        sum_data_num = 0  # データ数の合計
        for (inputs, labels) in trainloader:
            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_data_num += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        if lr_scheduling is not None:
            scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]
        train_loss = sum_loss*args.batch_size/len(trainloader.dataset)
        train_acc = float(sum_correct/sum_data_num)

        # 学習モデルのテスト
        sum_loss = 0.0
        sum_correct = 0
        sum_data_num = 0
        for (inputs, labels) in testloader:
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_data_num += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        test_loss = sum_loss*args.batch_size/len(testloader.dataset)
        test_acc = float(sum_correct/sum_data_n)

        print("{0:<13}{1:<13.5f}{2:<13.5f}{3:<13.5f}{4:<13.5f}{5:<6.6f}".format(epoch+1, train_loss, train_acc, test_loss, test_acc, lr))

    return net
