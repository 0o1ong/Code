import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim) 
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(0.2)
        
        # downsampling (in_dim != out_dim); identity mapping 바로 적용 불가
        if stride == 1: self.identity = nn.Sequential()  
        else: self.identity = nn.Conv2d(in_dim, dim, kernel_size=1, stride=stride, bias=False) # stride=2를 통해 feature map size 맞추기
            
    def forward(self, x):
        # full pre-activation: (BN -> ReLU -> conv -> BN -> ReLU -> conv) + x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        out += self.identity(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=10):
        super(ResNet, self).__init__()
        self.in_dim = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.residual_block(block, 16, block_num[0], first_stride=1)
        self.layer2 = self.residual_block(block, 32, block_num[1], first_stride=2)
        self.layer3 = self.residual_block(block, 64, block_num[2], first_stride=2)
        self.fc = nn.Linear(64, num_classes) # fc layer

    def residual_block(self, block, dim, num_blocks, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1) # stride가 맨 처음 conv 연산에만 적용되도록, 그 다음에는 전부 1(w, h 유지)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, dim, stride))
            self.in_dim = dim # dim update (다음 레이어 in_dim)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc(x), dim=1)
        return x

# ResNet56 함수 정의
def get_model():
    return ResNet(BasicBlock, [9, 9, 9]) # 각 레이어에 9개의 블록 사용
