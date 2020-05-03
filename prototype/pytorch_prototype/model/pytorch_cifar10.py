from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Cifar10_net(nn.Module):
    def __init__(self, out=3):
        super(Cifar10_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (1) 32*32*3 -> 28*28*6
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # (3) 14*14*6 -> 10*10*16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # (2) 28*28*6 -> 14*14*6
        x = self.pool(F.relu(self.conv2(x))) # (4) 10*10*16 -> 5*5*16
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
  def __init__(self, out=3):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # (1) 32*32*3 -> 32*32*16
    self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # (3) 16*16*16 -> 16*16*32
    self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1) # (5) 8*8*32 -> 8*8*64
    self.fc1 = nn.Linear(4*4*64, 500)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(500, out)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2) # (2) 32*32*16 -> 16*16*16
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2) # (4) 16*16*32 -> 8*8*32
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, 2, 2) # (6) 8*8*64 -> 4*4*64
    x = x.view(-1, 4*4*64)
    x = F.relu(self.fc1(x))
    feature = x
    x = self.dropout1(x)
    x = self.fc2(x)
    return x, feature
