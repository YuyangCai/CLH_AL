# dataset.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import get_config

Config = get_config()

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
        transforms.RandomHorizontalFlip(),     # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 的标准归一化参数
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    return train_transform, test_transform

def load_datasets(train_transform, test_transform):
    train_dataset = datasets.CIFAR10(
        root=Config.IMG_ROOT_TRAIN,          # 数据存放的根目录
        train=True,                     # 加载训练集
        download=True,                  # 如果数据未下载，则下载
        transform=train_transform        # 应用训练数据的变换
    )
    test_dataset = datasets.CIFAR10(
        root=Config.IMG_ROOT_TRAIN,          # 数据存放的根目录
        train=False,                    # 加载测试集
        download=True,                  # 如果数据未下载，则下载
        transform=test_transform          # 应用测试数据的变换
    )

    return train_dataset, test_dataset

def get_data_loaders(batch_size=128, num_workers=4):

    train_transform, test_transform = get_transforms()
    train_dataset, test_dataset = load_datasets(train_transform, test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
