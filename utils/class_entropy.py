# utils/entropy.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

#H(p)=-plog(p)-(1-p)log(1-p)

def calculate_entropy_multilabel_batch(probabilities):
    p = torch.clamp(probabilities, 1e-6, 1 - 1e-6)
    entropy_per_class = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    total_entropy = entropy_per_class.sum(dim=1)
    return total_entropy

def caculate_class_counts(probabilities,threshold = 0.5):
    positive_class_counts = (probabilities > threshold).sum(dim=1).float()
    return positive_class_counts
    

def calculate_entropy_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )   
    entropy_scores = np.empty(len(dataset), dtype=np.float32)
    total_batches = len(dataloader)    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Calculating Entropy Scores", unit="batch")):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)  
            probabilities = torch.sigmoid(outputs) 
            positive_class_counts = caculate_class_counts(probabilities,threshold = 0.5)  # Shape: (batch_size,) 
            entropy = calculate_entropy_multilabel_batch(probabilities)
            total_entropy = entropy * positive_class_counts
            start_idx = batch_idx * batch_size
            end_idx = start_idx + total_entropy.shape[0]
            entropy_scores[start_idx:end_idx] = total_entropy.cpu().numpy()
            
            del inputs, outputs, probabilities, total_entropy
            torch.cuda.empty_cache()
    
    return entropy_scores

def get_class_entropy_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):

    entropy_scores = calculate_entropy_scores(model, dataset,label_dataset, device, batch_size, num_workers)

    return entropy_scores
