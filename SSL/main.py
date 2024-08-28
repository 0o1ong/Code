import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from models import get_model, BasicBlock, PreActBlock, BottleNeck
from datasets import get_data_loaders
from train import train
from utils import log_setting, LinearWarmupCosineAnnealingLR, MLP, Module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='cifar10_resnet18_byol')
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train', type=str, default='byol')
    
    parser.add_argument('--version', type=str, default='v1') # for --train rotation (v1 / v2 / v3)
    
    args = parser.parse_args()  

    # Load Dataset
    ssl = ['simclr', 'moco', 'byol', 'simsiam']
    if args.train in ssl:
        train_loader, test_loader, pretrain_loader = get_data_loaders(args.batch_size, args.dataset, args.train)
    else:
        train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset, args.train)

    log_setting(args.logdir)
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    
    if args.train in ssl:
        if args.model == 'resnet':
            encoder = get_model(args.model, PreActBlock, [2, 2, 2, 2], 512).to(device)
        elif args.model == 'vit':
            encoder = get_model(args.model, out_dim=512).to(device)

        # Predictor
        if args.train in ['byol', 'sinsiam']:
            predictor = MLP()
        else:
            predictor = None
        online = Module.get_module(encoder, predictor)

        # Target network
        if args.train in ['moco', 'byol']:
            if args.model == 'resnet':
                target_encoder = get_model(args.model, PreActBlock, [2, 2, 2, 2], 512).to(device)
            elif args.model == 'vit':
                target_encoder = get_model(args.model, out_dim=512).to(device)
            target = Module.get_module(target_encoder, None)
        else:
            target = None
        optimizer = optim.AdamW(online.parameters(), weight_decay=0.05, lr=args.learning_rate)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=10, total_steps=args.epoch_num)
        train(args.train, online, target, train_loader, test_loader, pretrain_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir)
    else:
        if args.model == 'rotnet':
            model = get_model(args.model, PreActBlock, [2, 2, 2, 2], args.version).to(device)
            optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, momentum=0.9, lr=args.learning_rate)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
            train(args.train, model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir, args.version)
        else:    
            model = get_model(args.model).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=1e-5)
            train(args.train, model, train_loader, test_loader, optimizer, criterion, lr_scheduler, device, args.epoch_num, args.logdir)
    
if __name__ == '__main__': 
    main()
