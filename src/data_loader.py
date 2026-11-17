import pandas as pd
import numpy as np
import os
import tifffile
import cv2
try:
    from src.config import CROP_DATASET_PATH, HYPER_SPECTRAL_DIR, PEST_DATASET_DIR, IMAGE_SIZE
except ImportError:
    from config import CROP_DATASET_PATH, HYPER_SPECTRAL_DIR, PEST_DATASET_DIR, IMAGE_SIZE

def load_crop_data():
    return pd.read_csv(CROP_DATASET_PATH)

def load_hyperspectral_image(file_path):
    # tifffile for hyperspectral 16-bit
    return tifffile.imread(file_path)

def load_pest_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
    return img
