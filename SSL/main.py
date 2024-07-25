import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

import argparse

from models import get_model, BasicBlock, PreActBlock, BottleNeck
from datasets import get_data_loaders
from train import train

from utils import log_setting

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--logdir', type=str, default='log_moco') 
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train', type=str, default='moco')
    
    parser.add_argument('--version', type=str, default='v1') # for --train rotation (v1 / v2 / v3)
    
    args = parser.parse_args()

    log_setting(args.logdir)
    
    if args.train == 'simclr' or args.train == 'moco':
        train_loader, test_loader, pretrain_loader = get_data_loaders(args.batch_size, args.dataset, args.train)
    else:
        train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset, args.train)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    if args.model == 'rotnet':
        model = get_model(args.model, PreActBlock, [2, 2, 2, 2], args.version).to(device)
        optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, momentum=0.9, lr=args.learning_rate)
        train(args.train, model, train_loader, test_loader, optimizer, criterion, device, args.epoch_num, args.logdir, args.version)
    else:
        if args.train == 'simclr' or args.train == 'moco':
            model = get_model(args.model, BottleNeck, [3, 4, 6, 3]).to(device) # Base Encoder: ResNet-50
            optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=args.learning_rate, dampening=False)
            train(args.train, model, train_loader, test_loader, pretrain_loader, optimizer, device, args.epoch_num, args.logdir)
        else:
            model = get_model(args.model).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            train(args.train, model, train_loader, test_loader, optimizer, criterion, device, args.epoch_num, args.logdir)
        
if __name__ == '__main__': 
    main()
