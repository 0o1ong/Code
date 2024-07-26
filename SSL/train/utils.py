import torch
import torch.nn as nn
import os
import logging

# Extract Representation
def represent(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model.extract_features(inputs)
            features.append(outputs)
            labels.append(targets)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

# KNN accuracy
def KNN_acc(model, train_loader, test_loader, device):
    train_features, train_labels = represent(model, train_loader, device)
    test_features, test_labels = represent(model, test_loader, device)

    dist = torch.cdist(test_features, train_features, p=2)
    _, nearest_idx = dist.topk(k=1, dim=1, largest=False)
    nearest_idx = nearest_idx.squeeze().cpu()
    knn_predicted = train_labels[nearest_idx]

    knn_accuracy = ((knn_predicted == test_labels).float().mean()) * 100
    return knn_accuracy

# Linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Linear accuracy
def linear_acc(model, epoch, input_dim, num_classes, train_loader, test_loader, device):
    train_features, train_labels = represent(model, train_loader, device)
    test_features, test_labels = represent(model, test_loader, device)

    linear_classifier = LinearClassifier(input_dim, num_classes)
    linear_classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=0.01)

    linear_classifier.train()
    for _ in range(epoch):
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

def save_log(epoch, epoch_num, train_loss, **kwargs):
    log_message = f'Epoch [{epoch + 1}/{epoch_num}], Loss: {train_loss:.4f}'
    if 'val_loss' in kwargs:
        log_message += f', Val Loss: {kwargs["val_loss"]:.4f}'
    if 'val_acc' in kwargs:
        log_message += f', Val Accuracy: {kwargs["val_acc"]:.2f}%'
    if 'knn_acc' in kwargs:
        log_message += f', KNN Accuracy: {kwargs["knn_acc"]:.2f}%'
    logging.info(log_message)

def save_model(best_acc, current_acc, model, logdir, epoch):
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), os.path.join(logdir, 'best_model.pth'))
        logging.info(f'Checkpoint saved at epoch {epoch + 1} with accuracy {current_acc:.2f}%')
        return best_acc
