import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, stride=1):
        super(BasicBlock, self).__init__()

        # full pre-activation: (BN -> ReLU -> conv -> BN -> ReLU -> conv) + x
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        
        # downsampling (in_dim != out_dim); identity mapping 바로 적용 불가
        self.identity = nn.Sequential()  
        if stride != 1 or in_dim != dim:
            self.identity = nn.Conv2d(in_dim, dim, kernel_size=1, stride=stride, bias=False) # stride를 통해 feature map size 맞추기
            
    def forward(self, x):
        return self.residual(x) + self.identity(x)

class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=10):
        super(ResNet, self).__init__()
        self.in_dim = 16
        
        self.firstConv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer1 = self.residual_block(block, 16, block_num[0], first_stride=1)
        self.layer2 = self.residual_block(block, 32, block_num[1], first_stride=2)
        self.layer3 = self.residual_block(block, 64, block_num[2], first_stride=2)
        self.fc = nn.Linear(64, num_classes)

    def residual_block(self, block, dim, num_blocks, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, dim, stride))
            self.in_dim = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstConv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc(x), dim=1)
        return x

def get_model():
    return ResNet(BasicBlock, [18, 18, 18])
