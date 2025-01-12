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
            _, labels = batch  
            labels = labels.to(device)
            class_counts += labels.sum(dim=0)
    

    class_counts = class_counts.cpu().numpy()
    
    class_counts = np.maximum(class_counts, 1.0)
    

    class_weights = 1.0 / class_counts
    

    class_weights = class_weights / class_weights.sum()
    
    return class_weights

def calculate_diversity_scores(model, unlabel_dataset, class_weights, device, batch_size=256, num_workers=64):

    dataloader = DataLoader(
        unlabel_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    diversity_scores = np.empty(len(unlabel_dataset), dtype=np.float32)
    total_batches = len(dataloader)
    

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Calculating Class Scores", unit="batch")):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)  
            probabilities = torch.sigmoid(outputs)  # 形状: (batch_size, num_classes)
            

            threshold = 0.5  # 可以根据需求调整
            active_weights = (probabilities >= threshold).float() * class_weights_tensor  # 形状: (batch_size, num_classes)
            diversity = active_weights.sum(dim=1)  # 形状: (batch_size,)

            # 将多样性分数移动到 CPU 并转换为 NumPy
            diversity_scores_batch = diversity.cpu().numpy()
            
            start_idx = batch_idx * batch_size
            end_idx = start_idx + diversity_scores_batch.shape[0]
            diversity_scores[start_idx:end_idx] = diversity_scores_batch
            

            del inputs, outputs, probabilities, diversity
            torch.cuda.empty_cache()
    
    return diversity_scores

def get_class_scores(model, unlabel_dataset, label_dataset, device, num_classes,  
                                          batch_size=256, num_workers=64):

    # 计算类别权重
    class_weights = compute_class_weights(
        label_dataset=label_dataset,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )
    

    diversity_scores = calculate_diversity_scores(
        model=model,
        unlabel_dataset=unlabel_dataset,
        class_weights=class_weights,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return diversity_scores

