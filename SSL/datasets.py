from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size, dataset_name):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    if dataset_name == 'CIFAR10':
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader
