import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    def __init__(self, in_dim, dim, stride=1):
        super(PreActBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )
        self.identity = nn.Sequential()
        if stride != 1 or in_dim != dim:
            self.identity = nn.Conv2d(in_dim, dim, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.residual(x) + self.identity(x)

# Pre-train model
class ResNet(nn.Module):
    def __init__(self, block, block_num, version, num_classes=4):
        super(ResNet, self).__init__()
        self.in_dim = 64
        self.block = block
        self.version = version

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer1 = self.stage(block, 64, block_num[0], first_stride=1)
        self.layer2 = self.stage(block, 128, block_num[1], first_stride=2)
        self.layer3 = self.stage(block, 256, block_num[2], first_stride=2)
        self.layer4 = self.stage(block, 512, block_num[3], first_stride=2)
        self.bn = nn.BatchNorm2d(512)

        if self.version == 'v3':
            self.fc = nn.Sequential(nn.Linear(512, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_classes))
        else:
            self.fc = nn.Linear(512, num_classes)

    def stage(self, block, dim, num_blocks, first_stride):
        strides = [first_stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, dim, stride))
            self.in_dim = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1) # representations (512)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        return x
