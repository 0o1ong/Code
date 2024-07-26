import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from models import get_model, BasicBlock, PreActBlock, BottleNeck
from datasets import get_data_loaders
from train import train
from utils import log_setting, lr_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log') 
    parser.add_argument('--model', type=str, default='rotnet')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train', type=str, default='rotation')
    
    parser.add_argument('--version', type=str, default='v1') # for --train rotation (v1 / v2 / v3)
    
    args = parser.parse_args()  
    if args.train == 'simclr' or args.train == 'moco':
        train_loader, test_loader, pretrain_loader = get_data_loaders(args.batch_size, args.dataset, args.train)
    else:
        train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset, args.train)

    log_setting(args.logdir)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    if args.model == 'rotnet':
        model = get_model(args.model, PreActBlock, [2, 2, 2, 2], args.version).to(device)
        optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, momentum=0.9, lr=args.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[30, 60, 80],
                                                    gamma=0.2)
        train(args.train, model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir, args.version)
    else:
        if args.train == 'simclr' or args.train == 'moco':
            model = get_model(args.model, BottleNeck, [3, 4, 6, 3]).to(device) # Base Encoder: ResNet-50
            optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=args.learning_rate, dampening=False)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
            train(args.train, model, train_loader, test_loader, pretrain_loader, optimizer, lr_scheduler, device, args.epoch_num, args.logdir)
        else:
            model = get_model(args.model).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
            train(args.train, model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir)
        
if __name__ == '__main__': 
    main()
