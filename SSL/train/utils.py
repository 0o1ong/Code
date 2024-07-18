import torch
import torch.nn as nn

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

# KNN classifier
def KNNClassifier(train_features, train_labels, test_features):
    dist = torch.cdist(test_features, train_features, p=2)
    _, nearest_idx = dist.topk(k=1, dim=1, largest=False)
    nearest_idx = nearest_idx.squeeze().cpu()
    predicted = train_labels[nearest_idx]
    return predicted

# KNN accuracy
def KNN_acc(model, train_loader, test_loader, device):
    train_features, train_labels = represent(model, train_loader, device)
    test_features, test_labels = represent(model, test_loader, device)
    knn_predicted = KNNClassifier(train_features, train_labels, test_features)
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
def linear_acc(model, epoch, train_loader, test_loader, device):
    train_features, train_labels = represent(model, train_loader, device)
    test_features, test_labels = represent(model, test_loader, device)

    linear_classifier = LinearClassifier(512, 10)
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
    return linear_accuracy
