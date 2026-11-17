import numpy as np
import cv2

def extract_hyperspectral_features(img_arr):
    # Statistics per channel (125 channels)
    mean = np.mean(img_arr, axis=(0, 1))
    std = np.std(img_arr, axis=(0, 1))
    return np.concatenate([mean, std])

def extract_pest_features(img_arr):
    # Flatten RGB
    return img_arr.flatten()
