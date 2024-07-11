import torch
import torch.nn as nn
import torch.nn.functional as F

# Original
class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, stride=1):
        super(BasicBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.downsample = nn.Sequential(nn.Conv2d(in_dim, dim, kernel_size=1)) if in_dim != dim else nn.Sequential()
            
    def forward(self, x):
        return F.relu(self.residual(x) + self.downsample(x))

# Full pre-activation
class PreActBlock(nn.Module):
    def __init__(self, in_dim, dim):
        super(PreActBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.downsample = nn.Sequential(nn.Conv2d(in_dim, dim, kernel_size=1)) if in_dim != dim else nn.Sequential()

    def forward(self, x):
        return self.residual(x) + self.downsample(x)


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=10):
        super(ResNet, self).__init__()
        self.in_dim = 64
        self.block = block
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer1 = self.stage(block, 64, block_num[0])
        self.layer2 = self.stage(block, 128, block_num[1])
        self.layer3 = self.stage(block, 256, block_num[2])
        self.layer4 = self.stage(block, 512, block_num[3])
        self.bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes) 

    def stage(self, block, dim, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_dim, dim))
            self.in_dim = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = F.avg_pool2d(x, 2) # stride=2 -> pooling
        x = self.layer2(x)
        x = F.avg_pool2d(x, 2)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 2)
        x = self.layer4(x)
        # Pre-act의 경우 last residual block; extra activation
        if self.block == PreActBlock:
            x = self.bn(x)
            x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_model():
    return ResNet(PreActBlock, [2, 2, 2, 2]) 
