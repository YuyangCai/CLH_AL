import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.blur_detection import is_blurry
from utils.image_quality import check_image_quality
from config import get_config
from torch.utils.data import Subset
import cv2  # 确保 OpenCV 已导入

Config = get_config()

def get_indices_cifar10(dataset, n, uncertainty_scores, blur_threshold=100.0):
    selected_indices = []
    

    subset_indices = dataset.indices if isinstance(dataset, Subset) else list(range(len(dataset)))
    

    index_score = {idx: uncertainty_scores.get(idx, 0) for idx in subset_indices if idx < len(dataset.dataset)}
    

    sorted_indices = sorted(index_score.items(), key=lambda x: x[1], reverse=True)
    
    for idx, score in sorted_indices:
        if len(selected_indices) >= n:
            break  # 达到所需数量，退出循环
        
        try:
            # 获取图像和标签
            img, _ = dataset.dataset[idx] if isinstance(dataset, Subset) else dataset[idx]
            
            # # 如果图像是 Tensor 类型，则转换为 PIL.Image
            # if isinstance(img, torch.Tensor):
            #     img = transforms.ToPILImage()(img)
            
            # # 可选：检查图像是否模糊
            # if is_blurry(img, blur_threshold):
            #     continue  # 如果图像模糊，跳过该样本
            
            # # 可选：检查图像质量（假设 check_image_quality 返回布尔值）
            # if not check_image_quality(img):
            #     continue  # 如果图像质量不佳，跳过该样本
            
            # 如果图像通过所有检查，添加到选中的索引列表
            selected_indices.append(idx)
        
        except Exception as e:
            print(f"ERROR processing index {idx}: {e}")
            print(f"Dataset size: {len(dataset.dataset)}")  # 调试信息
            continue  # 出现错误时，跳过该样本
    
    print(f"Selected {len(selected_indices)} samples from the dataset")
    return selected_indices
