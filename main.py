import os
import argparse
import importlib
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epoch_num', type=int, default=300)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    model_module = importlib.import_module(args.model)
    dataset_module = importlib.import_module(args.dataset)

    model = model_module.get_model()
    train_loader, test_loader = dataset_module.get_data_loaders(args.batch_size)  
    train(model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir, args.model, args.dataset)

if __name__ == "__main__":
    main()
