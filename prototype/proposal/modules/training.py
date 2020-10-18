from collections import defaultdict
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, TypeVar


class TrainLogging:

    def __init__(self):
        self.log = []

    def stack(self, **kwargs):
        self.log.append(kwargs)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=4)


def process(trainloader, testloader, model, epochs: int, lr: float, lr_scheduling=None, log_savepath=None):

    log_dict = defaultdict(list)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if lr_scheduling is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduling)

    def train(trainloader) -> Tuple[float, float]:
        sum_loss, sum_correct, sum_dataN = 0.0, 0, 0
        for (inputs, labels) in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_dataN += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        train_loss = sum_loss*trainloader.batch_size/len(trainloader.dataset)
        train_acc = float(sum_correct/sum_dataN)
        return train_loss, train_acc

    def test(testloader) -> Tuple[float, float]:
        sum_loss, sum_correct, sum_dataN = 0.0, 0, 0
        for (inputs, labels) in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_dataN += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        test_loss = sum_loss*testloader.batch_size/len(testloader.dataset)
        test_acc = float(sum_correct/sum_dataN)
        return test_loss, test_acc

    print("\n{0:<13}{1:<13}{2:<13}{3:<13}{4:<13}{5:<6}".format("epoch","train/loss","train/acc","test/loss","test/acc","lr"))
    logging = TrainLogging()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(trainloader)
        test_loss, test_acc = test(testloader)
        lr = optimizer.param_groups[-1]["lr"]
        print("{0:<13}{1:<13.5f}{2:<13.5f}{3:<13.5f}{4:<13.5f}{5:<6.6f}".format(epoch, train_loss, train_acc, test_loss, test_acc, lr))
        logging.stack(epoch=epoch, train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc, lr=lr)
        if lr_scheduling is not None: scheduler.step()
    if log_savepath is not None:
        logging.save(log_savepath)

    return model
