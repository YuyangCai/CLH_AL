

import os
import numpy as np

def get_initial_subset(img_root, n, dataset):
    filename_to_index = {filename: idx for idx, (filename, _) in enumerate(dataset.img_labels)}    
    selected_indices = []    
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]
    
    if subdirs:
        # 如果存在子目录，按照原有逻辑处理
        for subdir in subdirs:
            subdir_path = os.path.join(img_root, subdir)
            all_images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]        
            if len(all_images) == 0:
                continue          
            selected_images = np.random.choice(all_images, size=min(n, len(all_images)), replace=False)        
            for img_name in selected_images:
                full_img_name = os.path.join(subdir, img_name)
                if full_img_name in filename_to_index:
                    selected_indices.append(filename_to_index[full_img_name])
                elif img_name in filename_to_index:
                    selected_indices.append(filename_to_index[img_name])
                else:
                    print(f"Warning: {full_img_name} couldn't find index in dataset.")
    else:
        # 如果没有子目录，直接在根目录下选取图片
        all_images = [f for f in os.listdir(img_root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if len(all_images) == 0:
            print("Warning: No images found in the root directory.")
            return selected_indices
        
        if n >= len(all_images):
            selected_images = all_images
        else:
            # 等间隔选取图片
            interval = len(all_images) / n
            selected_images = [all_images[int(i * interval)] for i in range(n)]
        
        for img_name in selected_images:
            if img_name in filename_to_index:
                selected_indices.append(filename_to_index[img_name])
            else:
                print(f"Warning: {img_name} couldn't find index in dataset.")
    
    return selected_indices