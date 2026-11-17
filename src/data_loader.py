import pandas as pd
import numpy as np
import os
import tifffile
from PIL import Image

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
    try:
        with Image.open(file_path) as image:
            return np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
    except (OSError, ValueError):
        return None
