import numpy as np

def normalize_tabular(df):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_norm = df.copy()
    for col in numeric_cols:
        std = df[col].std()
        if not np.isfinite(std) or std == 0:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df[col] - df[col].mean()) / std
    return df_norm

def preprocess_hyperspectral(image_arr):
    # 16-bit to 0-1 range
    return image_arr.astype(np.float32) / 65535.0

def preprocess_pest_image(img_arr):
    # RGB to 0-1
    return img_arr.astype(np.float32) / 255.0
