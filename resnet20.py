import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1 if in_channels == out_channels else 2, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1 ,1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 2)
        self.actv = F.relu
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.in_equal_out = in_channels == out_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        if self.in_equal_out:
            identity = x
        else:
            identity = self.maxpool(x)
            identity = F.pad(identity, (0, 0, 0, 0, self.out_channels // 4, self.out_channels // 4), "constant", 0)
        out = self.actv(self.bn1(self.conv1(x)))
        out = self.actv(torch.add(identity, self.bn2(self.conv2(out))))
        return out
        
class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualBlock(in_channels, out_channels))
        for i in range(n-1):
            self.blocks.append(ResidualBlock(out_channels, out_channels))
    
    def forward(self, x):
        for i, l in enumerate(self.blocks):
            x = l(x)
        return x
    
class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1 ,1, bias=False)
        self.layer1 = Layer(16, 16, 3)
        self.layer2 = Layer(16, 32, 3)
        self.layer3 = Layer(32, 64, 3)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)
        self.actv = F.relu
        self.bn = nn.BatchNorm2d(16)
        self._initialize_weights()
        
    def forward(self, x):
        out = self.actv(self.bn(self.conv1(x)))
        out = self.pool(self.layer3(self.layer2(self.layer1(out))))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)