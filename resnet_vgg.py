import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score,average_precision_score
from numpy import random
import random
from net.resnet18 import resnet18
random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
# 使用GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
import pickle
# 批次大小和训练轮数
batch_size = 256
num_epochs = 300


# 定义 CelebADataset 类（与之前相同）
class CelebADataset(Dataset):
    def __init__(self, img_root, label_file, transform=None):
        self.img_root = img_root
        self.transform = transform

        self.img_labels = []
        self.targets = []
        # 从.pkl文件加载标签

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
        labels = torch.tensor(labels, dtype=torch.float32)  # 直接转换为torch.tensor
        labels[labels == -1] = 0  # 将-1替换为0
        return image, labels

# 加载索引并创建 Subset
def create_subset_from_indices(dataset, indices):
    return Subset(dataset, indices)

# 使用示例
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 初始化完整数据集
full_dataset = CelebADataset(
    img_root='/home/cyy/A_studio/datasets/VGG-FACE2/VGG-FACE2/data/vggface2_train/vggface2_train/vggface2_train/train',
    label_file='/home/cyy/A_studio/datasets/VGG-FACE2/VGG-FACE2/data/train_new.pkl',
    transform=transform
)

test_dataset = CelebADataset(
    img_root='/home/cyy/A_studio/datasets/VGG-FACE2/VGG-FACE2/data/vggface2_test/vggface2_test/test',
    label_file='/home/cyy/A_studio/datasets/VGG-FACE2/VGG-FACE2/data/test_new.pkl',
    transform=test_transform
)

# indices = list(range(len(full_dataset)))
# random.shuffle(indices)
# # okboom = indices[:8631] #n1
# # okboom = indices[:17262] #n2
# # okboom = indices[:25893] #n3
# # okboom = indices[:34524] #n4
# # okboom = indices[:43155] #n5
# okboom = indices[:51786] #n6
# # okboom = indices[:60417] #n7
# # okboom = indices[:86310] #n10
# # okboom = indices[:129465] #n15
# dst_train = torch.utils.data.Subset(full_dataset, okboom)



# test_subset = create_subset_from_indices(test_dataset, okboom)

train_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
# train_dataloader = DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=32)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32)


# 验证函数
def val(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for img, labels in dataloader:
            img, labels = img.to(device), labels.to(device)
            output = model(img)
            loss = criterion(output, labels)
            val_loss += loss.item()

            predicted = (output > 0.5).float()  # 多标签分类的预测
            correct += (predicted == labels).float().sum().item()
            total += labels.numel()

    accuracy = 100.0 * correct / total
    return val_loss / len(dataloader), accuracy

# 训练函数
def train_model(train_dataloader, model, optimizer, criterion, test_dataloader=None, test_interval=10, num_epochs=100):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        for img, labels in train_dataloader:
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).float()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
        all_labels = np.concatenate(all_labels, axis=0)  
        all_predictions = np.concatenate(all_predictions, axis=0)

        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

        total = all_labels.size
        correct = (all_labels == all_predictions).sum()
        train_accuracy  = 100.0 * correct / total
        running_loss = running_loss / len(train_dataloader)
        
        print(f'Train Epoch {epoch + 1}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%, weighted_f1: {weighted_f1:.2f}')
        scheduler.step(running_loss)
        
        # test every test_interval
        if test_dataloader is not None and (epoch + 1) % test_interval == 0:
            test_accuracy, test_f1, mAP, Balanced_acc = test(model, test_dataloader)
            print(f'--- Test after Epoch {epoch + 1} ---')
            print(f'Test Accuracy: {test_accuracy:.2f}%, Test Weighted F1-score: {test_f1:.4f}')
            print(f'mAP: {mAP:.4f}')
            print(f'Balanced Accuracy: {Balanced_acc:.4f}')            



# 验证函数
def test(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    total = 0
    correct = 0
    with torch.no_grad():
        for img, labels in dataloader:
            img, labels = img.to(device), labels.to(device)
            output = model(img)

            predicted = (output > 0.5).float()  # 多标签分类的预测

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    mAP = 0.0
    num_labels = all_labels.shape[1]
    for i in range(num_labels):
        mAP += average_precision_score(all_labels[:, i], all_predictions[:, i])

    mAP /= num_labels

    "计算F1_score"
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted') 
    total = all_labels.size
    correct = (all_labels == all_predictions).sum()

    accuracy = 100.0 * correct / total
    # 矩阵化计算 Balanced Accuracy
    TP = np.sum((all_predictions == 1) & (all_labels == 1), axis=0)
    TN = np.sum((all_predictions == 0) & (all_labels == 0), axis=0)
    FP = np.sum((all_predictions == 1) & (all_labels == 0), axis=0)
    FN = np.sum((all_predictions == 0) & (all_labels == 1), axis=0)
    
    # 计算 Sensitivity 和 Specificity
    sensitivity = TP / (TP + FN + 1e-12)  # 避免除零
    specificity = TN / (TN + FP + 1e-12)
    
    # 计算 Balanced Accuracy
    balanced_acc_per_label = (sensitivity + specificity) / 2
    balanced_acc = np.mean(balanced_acc_per_label)    

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Weighted F1-score: {weighted_f1:.4f}')
    print(f'mAP: {mAP:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')

    return accuracy, weighted_f1, mAP, balanced_acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # set_seed(32)
    model =  resnet18(num_classes=47).to(device)
    print("网络参数总数: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)

    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失用于多标签分类
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    train_model(train_dataloader, model, optimizer, criterion,test_dataloader=test_dataloader, test_interval=10, num_epochs=100)

