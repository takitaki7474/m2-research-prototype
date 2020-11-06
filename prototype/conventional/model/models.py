import torch.nn as nn
import torch.nn.functional as F

class LeNet3(nn.Module):
  def __init__(self, out=3):
    super(LeNet3, self).__init__()
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
    x = self.dropout1(x)
    x = self.fc2(x)
    return x



class LeNet5(nn.Module):
  def __init__(self, out=5):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
    self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
    self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)
    self.conv6 = nn.Conv2d(128, 128, 3, 1, padding=1)
    self.fc1 = nn.Linear(4*4*128, 1000)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(1000, out)

  def forward(self, x):
    x = F.relu(self.conv1(x)) # 32*32*3 -> 32*32*16
    x = F.max_pool2d(x, 2, 2) # 32*32*16 -> 16*16*16

    x = F.relu(self.conv2(x)) # 16*16*16 -> 16*16*32
    x = F.max_pool2d(x, 2, 2) # 16*16*32 -> 8*8*32

    x = F.relu(self.conv3(x)) # 8*8*32 -> 8*8*64
    x = F.relu(self.conv4(x)) # 8*8*64 -> 8*8*64
    x = F.max_pool2d(x, 2, 2) # 8*8*64 -> 4*4*64

    x = F.relu(self.conv5(x)) # 4*4*64 -> 4*4*128
    x = F.relu(self.conv6(x)) # 4*4*128 -> 4*4*128

    x = x.view(-1, 4*4*128)
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)
    x = self.fc2(x)
    return x
