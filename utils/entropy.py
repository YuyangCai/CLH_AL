# utils/entropy.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def is_multi_label(target):
    """
    判断是否为多标签任务：
    - 如果 target 是一维张量（单个样本标签），或者
    - 如果 target 是二维张量（batch 标签矩阵）且标签取值为 0 或 1。
    """
    return (len(target.shape) == 1 or len(target.shape) == 2) and target.max().item() <= 1 and target.min().item() >= 0

def calculate_entropy_multilabel_batch(probabilities):
    """
    计算多标签分类任务的熵。
    H(p) = -p * log(p) - (1 - p) * log(1 - p) 对每个类别求和
    """
    p = torch.clamp(probabilities, 1e-6, 1 - 1e-6)  # 防止log(0)
    entropy_per_class = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    total_entropy = entropy_per_class.sum(dim=1)  # 对所有类别求和
    return total_entropy

def calculate_entropy_multiclass_batch(probabilities):
    """
    计算多分类任务的熵。
    H(p) = -sum(p_i * log(p_i)) 对每个样本的所有类别求和
    """
    p = torch.clamp(probabilities, 1e-6, 1 - 1e-6)  # 防止log(0)
    entropy_per_sample = - torch.sum(p * torch.log(p), dim=1)  # 对所有类别求和
    return entropy_per_sample

def calculate_entropy_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )   
    entropy_scores = np.empty(len(dataset), dtype=np.float32)
    
    is_ml = None  
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Calculating Entropy Scores", unit="batch")):
            if is_ml is None:
                is_ml = is_multi_label(targets)  # 根据第一个batch判断任务类型

            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)  # 获取模型输出

            if is_ml:
                # 多标签任务使用 Sigmoid 激活函数
                probabilities = torch.sigmoid(outputs)
                total_entropy = calculate_entropy_multilabel_batch(probabilities)
            else:
                # 多分类任务使用 Softmax 激活函数
                probabilities = torch.softmax(outputs, dim=1)
                total_entropy = calculate_entropy_multiclass_batch(probabilities)

            # 将当前批次的熵分数存入结果数组
            start_idx = batch_idx * batch_size
            end_idx = start_idx + total_entropy.shape[0]
            entropy_scores[start_idx:end_idx] = total_entropy.cpu().numpy()
            
            # 释放显存
            del inputs, outputs, probabilities, total_entropy
            torch.cuda.empty_cache()
    
    return entropy_scores

def get_entropy_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):
    """
    获取数据集的熵分数。
    
    参数:
    - model: 训练好的模型
    - dataset: 数据集
    - device: 计算设备
    - batch_size: 批量大小
    - num_workers: DataLoader 的工作线程数
    
    返回:
    - entropy_scores: Numpy 数组，包含每个样本的熵分数
    """
    entropy_scores = calculate_entropy_scores(model, dataset, label_dataset, device, batch_size, num_workers)
    return entropy_scores
