import os
import argparse

from models import *
from datasets import get_data_loaders
from train import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log_simclr')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train', type=str, default='simclr')
    
    parser.add_argument('--version', type=str, default='v1') # for --train rotation (v1 / v2 / v3)
    
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.logdir)):
        os.makedirs(os.path.join(args.logdir))

    train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset, args.train)

    if args.model == 'rotnet':
        model = get_model(args.model, PreActBlock, [2, 2, 2, 2], args.version)
        train(args.train, model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir, args.version)
    else:
        if args.train == 'simclr' or args.train == 'moco':
            model = get_model(args.model, BottleNeck, [3, 4, 6, 3]) # Base Encoder: ResNet-50
            train(args.train, model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir, args.batch_size)
        else:
            model = get_model(args.model)
            train(args.train, model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir)
        
if __name__ == '__main__': 
    main()
