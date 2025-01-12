# utils/index_selection.py
import os
from PIL import Image
import numpy as np
from utils.blur_detection import is_blurry
from utils.image_quality import check_image_quality
from config import get_config

Config = get_config()
from torch.utils.data import Subset

def get_indices_per_subfolder(img_root, n, dataset, uncertainty_scores, blur_threshold=100.0):
    """
    根据数据集结构选择图片索引。
    
    如果数据集由子文件夹下的图片构成，则按子文件夹处理。
    如果目录下直接是图片，则等间隔划分区域，并在每个区域内选择不确定性最高的图片。
    
    参数:
        img_root (str): 图片根目录路径。
        n (int): 每个子文件夹或每个区域选择的图片数量。
        dataset (Dataset or Subset): 数据集对象。
        uncertainty_scores (dict): 不确定性评分，键为索引，值为评分。
        blur_threshold (float): 模糊检测阈值。
    
    返回:
        selected_indices (list): 选择的图片索引列表。
    """
    filter_type = Config.FILTER_TYPE
    
    # 获取原始数据集
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
    else:
        original_dataset = dataset    
        
    # 创建文件名到索引的映射
    filename_to_index = {filename: idx for idx, (filename, _) in enumerate(original_dataset.img_labels)}
    selected_indices = []
    
    # 检查是否有子文件夹
    subdirs = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]
    
    if subdirs:
        # 数据集由子文件夹构成，使用现有的逻辑
        for subdir in subdirs:
            subdir_path = os.path.join(img_root, subdir)
            
            all_images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
            
            if not all_images:
                continue  
            
            image_score = {}
            for img_name in all_images:
                full_img_name = os.path.join(subdir, img_name)
                idx = filename_to_index.get(full_img_name, None)
                if idx is not None:
                    score = uncertainty_scores.get(idx, None)  # 使用字典的 get 方法
                    if score is not None:
                        image_score[img_name] = score
            
            if not image_score:
                continue
            
            # 按不确定性评分降序排序
            sorted_images = sorted(image_score.items(), key=lambda x: x[1], reverse=True)
            
            selected_images = []
            
            for img_name, _ in sorted_images:
                img_path = os.path.join(subdir_path, img_name)
                # try:  
                #     if filter_type == 'is_blurry':
                #         if is_blurry(img_path, threshold=blur_threshold):
                #             continue
                #     elif filter_type == 'check_image_quality':
                #         if check_image_quality(
                #             img_path,
                #             blur_threshold=blur_threshold,
                #             dark_threshold=30,
                #             bright_threshold=220,
                #             color_variance_threshold=5,
                #             min_aspect_ratio=0.2,
                #             max_aspect_ratio=5.0
                #         ):
                #             continue
                    
                #     with Image.open(img_path) as img:
                #         width, height = img.size
                    
                #     if width < 80 or height < 80:
                #         continue  
                    
                selected_images.append(img_name)
                
                if len(selected_images) >= n:
                    break

                # except Exception as e:
                #     print(f"ERROR {img_path} ERROR: {e}")
                #     continue  
            
            for img_name in selected_images:
                full_img_name = os.path.join(subdir, img_name)
                if full_img_name in filename_to_index:
                    selected_indices.append(filename_to_index[full_img_name])
                else:
                    print(f"Couldn't Find index of: {full_img_name}")
    else:
        # 目录下直接是图片，等间隔划分区域并选择每个区域内不确定性最高的图片
        all_images = [f for f in os.listdir(img_root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        
        if not all_images:
            return selected_indices  # 空目录
        
        # 按文件名排序以保证一致性
        all_images.sort()
        total_images = len(all_images)
        
        if n <= 0:
            print("Parameter 'n' must be greater than 0 for flat directory.")
            return selected_indices
        
        # 计算每个区域的大小
        region_size = max(total_images // n, 1)
        
        for i in range(n):
            start_idx = i * region_size
            end_idx = min((i + 1) * region_size, total_images)
            region_images = all_images[start_idx:end_idx]
            
            if not region_images:
                continue
            
            # 在当前区域内找到不确定性最高的图片
            max_score = -np.inf
            selected_img = None
            for img_name in region_images:
                idx = filename_to_index.get(img_name, None)
                if idx is not None:
                    score = uncertainty_scores.get(idx, None)
                    if score is not None and score > max_score:
                        max_score = score
                        selected_img = img_name
            
            if selected_img:
                img_path = os.path.join(img_root, selected_img)
                try:
                    if filter_type == 'is_blurry':
                        if is_blurry(img_path, threshold=blur_threshold):
                            continue
                    elif filter_type == 'check_image_quality':
                        if check_image_quality(
                            img_path,
                            blur_threshold=blur_threshold,
                            dark_threshold=30,
                            bright_threshold=220,
                            color_variance_threshold=5,
                            min_aspect_ratio=0.2,
                            max_aspect_ratio=5.0
                        ):
                            continue
                    
                    with Image.open(img_path) as img:
                        width, height = img.size
                    
                    if width < 80 or height < 80:
                        continue  
                    
                    if len(selected_indices) >= n:
                        break  # 已经选择了足够的图片
                    
                    if selected_img in filename_to_index:
                        selected_indices.append(filename_to_index[selected_img])
                    else:
                        print(f"Couldn't Find index of: {selected_img}")
                
                except Exception as e:
                    print(f"ERROR {img_path} ERROR: {e}")
                    continue
    
    return selected_indices