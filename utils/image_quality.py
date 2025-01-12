import cv2
import numpy as np

def check_image_quality(
    image,
    blur_threshold=10,
    dark_threshold=30,
    bright_threshold=220,
    color_variance_threshold=5,
    min_aspect_ratio=0.2,
    max_aspect_ratio=5.0
):
    if image is None:
        raise ValueError("Image is None")

    # 1. 获取图像分辨率
    h, w = image.shape[:2]

    # 2. 判断长宽比
    if h == 0 or w == 0:
        return True  # 不合格
    
    ratio = w / h
    if ratio < min_aspect_ratio or ratio > max_aspect_ratio:
        return True  # 不合格

    # 3. 判断是否过暗或过亮
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val < dark_threshold or mean_val > bright_threshold:
        return True  # 不合格

    # 4. 判断是否几乎纯色
    b_var = np.var(image[:, :, 0])
    g_var = np.var(image[:, :, 1])
    r_var = np.var(image[:, :, 2])
    if b_var < color_variance_threshold and g_var < color_variance_threshold and r_var < color_variance_threshold:
        return True  # 不合格

    # 5. 判断是否模糊
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    if variance < blur_threshold:
        return True  # 不合格

    return False  # 合格
