import torch
import torchvision
from torchvision import datasets, transforms

Dataset_path = '/Datasets/CIFAR10/'

def get_cifar10_loaders(batch_size=256, num_workers=8, data_path="./data"):
    # 데이터 전처리
    transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 데이터셋
    Train_data = datasets.CIFAR10(root=Dataset_path, train=True, download=True, transform=transform_train)
    Test_data = datasets.CIFAR10(root=Dataset_path,  train=False, download=True, transform=transform_test)


    # DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        dataset=Train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=Test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )

    return train_data_loader, test_data_loader
