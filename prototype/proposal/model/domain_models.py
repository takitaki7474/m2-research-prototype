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
    feature = x
    x = self.dropout1(x)
    x = self.fc2(x)
    return x, feature
