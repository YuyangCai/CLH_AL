import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def compute_class_weights(label_dataset, num_classes, batch_size=256, num_workers=64, device='cpu'):

    dataloader = DataLoader(
        label_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_counts = torch.zeros(num_classes, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch[1]  # 假设标签在第二个元素
            labels = labels.to(device)
            class_counts += torch.bincount(labels, minlength=num_classes).float()
    
    class_counts = class_counts.cpu().numpy()
    class_counts = np.maximum(class_counts, 1.0)  # 防止除零
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # 归一化
    
    return class_weights

def calculate_combined_scores(model, dataset, class_weights, device, batch_size=256, num_workers=64):

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    final_scores = np.empty(len(dataset), dtype=np.float32)
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Calculating Final Scores", unit="batch")):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)  # 获取模型输出
            
            probabilities = torch.softmax(outputs, dim=1)  # 应用 Softmax 激活函数
            
            # 计算 Margin 分数
            sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
            p1 = sorted_probs[:, 0]  # 第一高概率
            p2 = sorted_probs[:, 1]  # 第二高概率
            margin = p1 - p2
            margin = -margin  # 计算 Margin
            
            # 获取预测类别的权重
            _, preds = torch.max(probabilities, dim=1)  # preds: (batch_size,)
            weights = class_weights_tensor[preds]  # (batch_size,)
            
            # 计算最终得分
            final_score = margin * weights  # Element-wise multiplication
            
            # 将当前批次的最终得分存入结果数组
            final_scores_batch = final_score.cpu().numpy()
            
            start_idx = batch_idx * batch_size
            end_idx = start_idx + final_scores_batch.shape[0]
            final_scores[start_idx:end_idx] = final_scores_batch
            
            # 释放显存
            del inputs, outputs, probabilities, sorted_probs, p1, p2, margin, preds, weights, final_score
            torch.cuda.empty_cache()
    
    return final_scores

def get_class_margin(model, unlabel_dataset, label_dataset, device, num_classes,  
                        batch_size=256, num_workers=64):

    # 计算类别权重
    
    class_weights = compute_class_weights(
        label_dataset=label_dataset,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )
    
    # 计算最终得分
    final_scores = calculate_combined_scores(
        model=model,
        dataset=unlabel_dataset,
        class_weights=class_weights,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return final_scores
