import os
import argparse

from models import *
from datasets import get_data_loaders
from train import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log_vit')
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train', type=str, default='basic')
    parser.add_argument('--version', type=str, default='v1') # for --train rotation (v1 / v2 / v3)
    
    args = parser.parse_args()

    if args.model == 'rotnet':
        if not os.path.exists(os.path.join(args.logdir, args.version)):
            os.makedirs(os.path.join(args.logdir, args.version))
    else:
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)

    train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset)

    if args.model == 'rotnet':
        model = get_model(args.model, PreActBlock, [2, 2, 2, 2], args.version)
        train(args.train, model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir, args.version)
    else:
        model = get_model(args.model)
        train(args.train, model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir)
        
if __name__ == '__main__': 
    main()
