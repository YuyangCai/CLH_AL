# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle
from torchvision import transforms
from config import Config

class VGGDataset(Dataset):
    def __init__(self, img_root, label_file, transform=None):
        self.img_root = img_root
        self.transform = transform

        self.img_labels = []
        self.targets = []
        
        with open(label_file, 'rb') as f:
            df = pickle.load(f)  # 假设这是一个DataFrame
            self.img_labels = list(zip(df['Filename'], df.iloc[:, 2:].values))  # 提取文件名和标签
            self.targets = df.iloc[:, 2:].values.tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, labels = self.img_labels[idx]
        img_path = os.path.join(self.img_root, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)  # 直接转换为 torch.tensor
        labels[labels == -1] = 0  # 将 -1 替换为 0
        return image, labels
    

def get_transforms():
    """
    定义训练和测试的数据增强和预处理变换。

    Returns:
        tuple: (训练变换, 测试变换)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return train_transform, test_transform

def load_datasets(train_transform, test_transform):

    full_dataset = VGGDataset(
        img_root=Config.IMG_ROOT_TRAIN,
        label_file=Config.LABEL_FILE_TRAIN,
        transform=train_transform
    )
    test_dataset = VGGDataset(
        img_root=Config.IMG_ROOT_TEST,
        label_file=Config.LABEL_FILE_TEST,
        transform=test_transform
    )
    return full_dataset, test_dataset