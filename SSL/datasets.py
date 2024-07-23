from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Positive Pair Aug
class DataAug():
    def __init__(self, dataset):
        self.dataset = dataset # train_loader
        self.transform = transforms.Compose([
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=0.5, std=0.5)
                        ])
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): 
        img, label = self.dataset[idx]
        img1 = self.transform(img) # 1st aug
        img2 = self.transform(img) # 2nd aug
        return img1, img2, label # positive pair

def get_data_loaders(batch_size, dataset_name, train):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform if train!='simclr' else None)
        test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if train == 'simclr':
        aug_train_set = DataAug(train_set)
        train_loader = DataLoader(aug_train_set, batch_size=batch_size, shuffle=True)
    else:
        train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader
