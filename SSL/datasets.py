from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Positive Pair Aug
class SimCLRAug():
    def __init__(self, t):
        self.t = t

    def __call__(self, x): 
        x1 = self.t(x) # 1st aug
        x2 = self.t(x) # 2nd aug
        return x1, x2 # positive pair

def get_data_loaders(batch_size, dataset_name, train_mode):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    ssl = ['simclr', 'moco', 'byol', 'simsiam']

    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform if train_mode not in ssl else test_transform)
        test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform if train_mode not in ssl else test_transform)
        if train_mode in ssl:
            aug_train_set = datasets.CIFAR10('./data', train=True, download=True, transform=SimCLRAug(aug_transform))
    elif dataset_name == 'cifar100':
        train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform if train_mode not in ssl else test_transform)
        test_set = datasets.CIFAR100('./data', train=False, download=True, transform=transform if train_mode not in ssl else test_transform)
        if train_mode in ssl:
            aug_train_set = datasets.CIFAR100('./data', train=True, download=True, transform=SimCLRAug(aug_transform))
    elif dataset_name == 'stl10':
        train_set = datasets.STL10('./data', split='train', download=True, transform=transform if train_mode not in ssl else test_transform)
        test_set = datasets.STL10('./data', split='test', download=True, transform=transform if train_mode not in ssl else test_transform)
        if train_mode in ssl:
            aug_train_set = datasets.STL10('./data', split='train', download=True, transform=SimCLRAug(aug_transform))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True if train_mode not in ssl else False) 
    test_loader = DataLoader(test_set, batch_size=batch_size)

    if train_mode in ssl:
        pretrain_loader = DataLoader(aug_train_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, pretrain_loader
    else:
        return train_loader, test_loader
