import torch.nn as nn
import torch.nn.functional as F

class PreLeNet(nn.Module):
    def __init__(self, out=3):
        super(PreLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # (1) 32*32*3 -> 32*32*16
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # (3) 16*16*16 -> 16*16*32
        self.gap = nn.AvgPool2d(kernel_size=8)
        self.fc1 = nn.Linear(8*8*32, out)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2) # (2) 32*32*16 -> 16*16*16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2) # (4) 16*16*32 -> 8*8*32
        feature = self.gap(x) # 1*1*32
        feature = feature.view(-1, 32) # 1*32
        x = x.view(-1, 8*8*32)
        x = F.relu(self.fc1(x))
        return x, feature
