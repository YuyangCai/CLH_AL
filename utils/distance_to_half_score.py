import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def calculate_distance_to_half_multilabel_batch(probabilities):
    distance = torch.abs(probabilities - 0.5)
    distance_to_half = distance.sum(dim=1)
    return distance_to_half

def calculate_distance_to_half_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )   
    distance_to_half_scores = np.empty(len(dataset), dtype=np.float32)
    total_batches = len(dataloader)    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Calculating Distance to Half Scores", unit="batch")):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)  
            probabilities = torch.sigmoid(outputs) 

            distance_to_half = calculate_distance_to_half_multilabel_batch(probabilities)  # caculate distance_to_half
            start_idx = batch_idx * batch_size
            end_idx = start_idx + distance_to_half.shape[0]
            distance_to_half_scores[start_idx:end_idx] = distance_to_half.cpu().numpy()
            
            del inputs, outputs, probabilities, distance_to_half
            torch.cuda.empty_cache()
    
    return distance_to_half_scores

def get_distance_to_half_scores(model, dataset,label_dataset, device, batch_size=256, num_workers=64):

    distance_to_half_scores = calculate_distance_to_half_scores(model, dataset,label_dataset, device, batch_size, num_workers)
    return distance_to_half_scores
