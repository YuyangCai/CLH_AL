# utils/margin.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def calculate_margin_multiclass_batch(probabilities):

    sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
    p1 = sorted_probs[:, 0]  # 第一高概率
    p2 = sorted_probs[:, 1]  # 第二高概率
    margin = p1 - p2
    margin = -margin          # 计算 Margin
    return margin

def calculate_margin_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )   
    margin_scores = np.empty(len(dataset), dtype=np.float32)
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Calculating Margin Scores", unit="batch")):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)  # 获取模型输出
            
            probabilities = torch.softmax(outputs, dim=1)  # 应用 Softmax 激活函数
            margins = calculate_margin_multiclass_batch(probabilities)  # 计算 Margin
            
            # 将当前批次的 Margin 分数存入结果数组
            start_idx = batch_idx * batch_size
            end_idx = start_idx + margins.shape[0]
            margin_scores[start_idx:end_idx] = margins.cpu().numpy()
            
            # 释放显存
            del inputs, outputs, probabilities, margins
            torch.cuda.empty_cache()
    
    return margin_scores

def get_margin_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):
    """
    获取数据集的 Margin 分数，仅适用于多分类任务。
    
    参数:
    - model: 训练好的模型
    - dataset: 数据集
    - device: 计算设备
    - batch_size: 批量大小
    - num_workers: DataLoader 的工作线程数
    
    返回:
    - margin_scores: Numpy 数组，包含每个样本的 Margin 分数
    """
    margin_scores = calculate_margin_scores(model, dataset,label_dataset,device, batch_size, num_workers)
    return margin_scores
