# utils/blur_detection.py
import cv2
import os
from PIL import Image

def calculate_laplacian_variance(image_path):
 
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Couldn't read img : {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)


    variance = laplacian.var()

    return variance

def is_blurry(image_path, threshold=100.0):

    variance = calculate_laplacian_variance(image_path)

    return variance < threshold
