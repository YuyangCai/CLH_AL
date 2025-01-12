import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
from net.resnet18 import resnet18  # 确保实现了 ResNet18
from net.resnet18_cifar import resnet18_cifar
from torchvision import transforms as T
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 硬件设置
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 批次大小和训练轮数
batch_size = 128
num_epochs = 200
cifar10_path = "/home/cyy/A_studio/datasets/cifar10"  # CIFAR-10 数据集路径
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
# 数据预处理
transform_train = transforms.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])

transform_test = transforms.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

# 加载 CIFAR-10 数据集
full_train_dataset = torchvision.datasets.CIFAR10(
    root=cifar10_path, train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root=cifar10_path, train=False, download=True, transform=transform_test
)

# 随机选择部分样本用于训练
def create_subset(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    return Subset(dataset, subset_indices)

# 设置训练样本数量（可以自定义）
num_train_samples = 50000  
train_dataset = create_subset(full_train_dataset, num_train_samples)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 验证函数
def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)  # 获取每个样本的预测类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    # 输出分类报告
    class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print("Validation Classification Report:\n", class_report)
    return val_loss, accuracy, f1

# 训练函数
def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        all_preds = []
        all_labels = []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)  # 获取每个样本的预测类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")

        # 验证
        val_loss, val_acc, val_f1 = validate(model, test_loader, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # 更新学习率
        scheduler.step()

# 主函数
if __name__ == "__main__":
    set_seed(123)
    model = resnet18_cifar(num_classes=10).to(device)  # 适配 CIFAR-10 的 10 个类别
    print(f"网络参数总数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov = True)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-4
    )

    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=num_epochs)
