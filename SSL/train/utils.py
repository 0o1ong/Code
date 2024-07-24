import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

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
    return linear_accuracy

def aug(inputs):
    aug_img = []
    aug_trans = transforms.Compose([
                  transforms.RandomResizedCrop(32),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                  transforms.RandomGrayscale(p=0.2),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=0.5, std=0.5)
                  ])
    
    for img in inputs:
        pil_img = to_pil_image(img)  # Tensor -> PIL
        aug_img.append(aug_trans(pil_img))
    return torch.stack(aug_img)

def NT_Xent(z, temperature, device): # z.size(): (2batch_size, 512)
    '''sim = torch.matmul(z, z.T) 
    norms = torch.norm(z, dim=1, keepdim=True) # 각 z의 norm값
    norms_prodict = torch.matmul(norms, norms.T) # ||z_i||*||z_j||, size: (2N, 2N)
    cos_sim = sim / norms_prodict
    cos_sim /= temperature'''

    z = F.normalize(z, dim=1)
    cos_sim = torch.matmul(z, z.T) / temperature

    indices = torch.arange(cos_sim.size(0))
    cos_sim[indices, indices] = float('-inf') # exp(-inf)==0, 같은 이미지에 대한 유사도(sim_{i, i}) 무시 가능
    # 각 Column마다 target은 positive pair (2k -> 2k-1 , 2k-1 -> 2k)
    target = torch.arange(cos_sim.size(0)).to(device)
    for i in range(cos_sim.size(0)):
        if (i % 2) == 0: # 2k
            target[i] += 1
        else:            # 2k-1
            target[i] -= 1
    return F.cross_entropy(cos_sim, target)
