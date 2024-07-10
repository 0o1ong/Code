from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])

    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader
