import os
import torch
import logging
import argparse

from model import ResNet, PreActBlock, FeatureExtractor, extract_features, LinearClassifier, KNNClassifier
from datasets import get_data_loaders
from train import train

def main(args):
    if not os.path.exists(os.path.join(args.logdir, args.version)):
        os.makedirs(os.path.join(args.logdir, args.version))

    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(os.path.join(args.logdir, args.version, 'training.log')),
        logging.StreamHandler()
    ])
    
    pre_train_model = ResNet(PreActBlock, [2, 2, 2, 2])
    train_loader, test_loader = get_data_loaders(args.batch_size, args.dataset)
    
    train(pre_train_model, train_loader, test_loader, args.epoch_num, args.learning_rate, args.logdir, args.version)

    transfered_model = FeatureExtractor(PreActBlock, [2, 2, 2, 2])
    pretrained_dict = torch.load(os.path.join(args.logdir, args.version, 'best_model.pth'))
    model_dict = transfered_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    transfered_model.load_state_dict(model_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transfered_model.to(device)

    train_features, train_labels = extract_features(transfered_model, train_loader, device)
    test_features, test_labels = extract_features(transfered_model, test_loader, device)

    knn_predicted = KNNClassifier(train_features, train_labels, test_features)
    knn_accuracy = ((knn_predicted == test_labels).float().mean()) * 100
    logging.info(f"KNN Accuracy: {knn_accuracy:.2f}%")

    linear_classifier = LinearClassifier(512, 10, args.projection, args.projection_dim)
    linear_classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=0.01)

    linear_classifier.train()
    for epoch in range(args.epoch_num):
        optimizer.zero_grad()
        outputs = linear_classifier(train_features).cpu()
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
    linear_classifier.eval()
    with torch.no_grad():
        outputs = linear_classifier(test_features)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()

    linear_accuracy = ((predicted == test_labels).float().mean()) * 100
    logging.info(f"Linear Accuracy: {linear_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='final_log')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--version', type=str, required=True) # v1 / v2
    parser.add_argument('--projection', type=bool, required=True) # 추출된 feature을 분류하기 전 MLP 적용 여부
    parser.add_argument('--projection_dim', type=int, default=512)
    args = parser.parse_args()

    main(args)
