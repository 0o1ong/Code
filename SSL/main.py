import argparse

from model import PreActBlock, ResNet
from datasets import get_data_loaders
from train import train

def main(args):

    pre_train_model = ResNet(PreActBlock, [2, 2, 2, 2], args.version)
    train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset)
    
    train(pre_train_model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir, args.version)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epoch_num', type=int, default=3)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--version', type=str, default='v3') # v1 / v2 / v3
    args = parser.parse_args()

    main(args)
