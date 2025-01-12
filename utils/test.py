# test.py
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score,average_precision_score

from config import get_config
Config = get_config()

device = Config.DEVICE

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

            predicted = (torch.sigmoid(output) > 0.5).float()  # 多标签分类的预测

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    mAP = 0.0
    num_labels = all_labels.shape[1]
    for i in range(num_labels):
        mAP += average_precision_score(all_labels[:, i], all_predictions[:, i])
    mAP /= num_labels
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted') 
    total = all_labels.size
    correct = (all_labels == all_predictions).sum()

    accuracy = 100.0 * correct / total
    
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
    print(f'mAP: {mAP:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Weighted F1-score: {weighted_f1:.4f}')

    return accuracy, weighted_f1,mAP,balanced_acc