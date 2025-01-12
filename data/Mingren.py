import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from config import get_config
Config = get_config()
import warnings

# 忽略UserWarning类型的警告
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

class MingrenDataset(Dataset):
    def __init__(self, img_root, label_file, transform=None):
        """
        初始化数据集
        :param img_root: 图片文件夹的根目录
        :param label_file: 标签文件的路径 (csv 文件)
        :param transform: 图像数据增强或预处理
        """
        self.img_root = img_root
        self.transform = transform
        self.img_labels = []
        self.targets = []
        self.classes = [
            "Male", "Pale_Skin", "Bald", "Receding_Hairline", "Eyeglasses", 
            "Big_Nose", "Pointy_Nose", "Rosy_Cheeks", "Big_Lips", "Wearing_Lipstick"
        ]  # 标签对应的类别名称

        # 加载 CSV 文件
        label_data = np.loadtxt(label_file, delimiter=",", dtype=np.int32)

        # 第一行是子文件夹编号
        self.subfolders = label_data[0]  # 每一列表示一个子文件夹
        # 每列标签代表该子文件夹中所有图片的标签
        self.targets = label_data[1:].T.tolist()  # 转置后每行对应一个图片的标签

        # 遍历子文件夹，构建图片路径列表和对应标签
        for folder_idx, folder_id in enumerate(self.subfolders):
            folder_path = os.path.join(self.img_root, str(folder_id))
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"文件夹 {folder_path} 不存在！")

            # 获取文件夹下所有图片文件
            images = sorted(os.listdir(folder_path))  # 确保图片顺序一致
            for img_name in images:
                img_path = os.path.join(str(folder_id), img_name)  # 只存相对路径
                self.img_labels.append((img_path, self.targets[folder_idx]))

    def __len__(self):
        """
        返回数据集大小
        """
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

    full_dataset = MingrenDataset(
        img_root=Config.IMG_ROOT_TRAIN,
        label_file=Config.LABEL_FILE_TRAIN,
        transform=train_transform
    )
    test_dataset = MingrenDataset(
        img_root=Config.IMG_ROOT_TEST,
        label_file=Config.LABEL_FILE_TEST,
        transform=test_transform
    )
    return full_dataset, test_dataset
