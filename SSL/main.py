import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from models import get_model, BasicBlock, PreActBlock, BottleNeck
from datasets import get_data_loaders
from train import train
from utils import log_setting, LinearWarmupCosineAnnealingLR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--logdir', type=str, default='log_moco') 
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train', type=str, default='moco')
    
    parser.add_argument('--version', type=str, default='v1') # for --train rotation (v1 / v2 / v3)
    
    args = parser.parse_args()  
    if args.train == 'simclr' or args.train == 'moco':
        train_loader, test_loader, pretrain_loader = get_data_loaders(args.batch_size, args.dataset, args.train)
    else:
        train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset, args.train)

    log_setting(args.logdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    if args.model == 'rotnet':
        model = get_model(args.model, PreActBlock, [2, 2, 2, 2], args.version).to(device)
        optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, momentum=0.9, lr=args.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[30, 60, 80],
                                                    gamma=0.2)
        train(args.train, model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir, args.version)
    else:
        if args.train == 'simclr':
            model = get_model(args.model, BottleNeck, [3, 4, 6, 3]).to(device) # Base Encoder: ResNet-50
            optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, nesterov=True, lr=args.learning_rate, dampening=False)
            lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=10, total_steps=args.epoch_num)
            train(args.train, model, train_loader, test_loader, pretrain_loader, optimizer, lr_scheduler, device, args.epoch_num, args.logdir)
        elif args.train == 'moco':
            # batch_size = 256, epoch_num=200 
            f_q = get_model(args.model, BottleNeck, [3, 4, 6, 3], 128).to(device) # Query Encoder
            f_k = get_model(args.model, BottleNeck, [3, 4, 6, 3], 128).to(device) # Key Encoder
            optimizer = optim.SGD(f_q.parameters(), weight_decay=1e-4, momentum=0.9, lr=args.learning_rate)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                  milestones=[120, 160], 
                                                  gamma=0.1)
            train(args.train, f_q, f_k, train_loader, test_loader, pretrain_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir)
        else:
            model = get_model(args.model).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)
            # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1) # for vit
            train(args.train, model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir)
        
if __name__ == '__main__': 
    main()
