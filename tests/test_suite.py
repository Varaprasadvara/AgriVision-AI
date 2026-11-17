import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from data_loader import load_crop_data, load_hyperspectral_image, load_pest_image
from preprocessing import normalize_tabular, preprocess_hyperspectral, preprocess_pest_image
from feature_extraction import extract_hyperspectral_features, extract_pest_features
from unsupervised_learning import run_pca, run_kmeans, run_dbscan, run_anomaly_detection
from config import CROP_DATASET_PATH, HYPER_SPECTRAL_DIR, PEST_DATASET_DIR

def test_crop_module():
    print("Testing Crop Module...")
    df = load_crop_data()
    assert not df.empty, "Crop data is empty"
    numeric_df = df.select_dtypes(include=[np.number])
    norm_df = normalize_tabular(numeric_df)
    assert norm_df.shape == numeric_df.shape, "Normalization failed"

    pca_res = run_pca(norm_df.values)
    assert pca_res.shape[1] == 2, "PCA failed"

    clusters = run_kmeans(norm_df.values, n_clusters=3)
    assert len(clusters) == len(df), "K-Means failed"

    anomalies = run_anomaly_detection(norm_df.values)
    assert len(anomalies) == len(df), "Isolation Forest failed"
    print("Crop Module: PASS")

def test_spectral_module():
    print("Testing Spectral Module...")
    import glob
    files = glob.glob(os.path.join(HYPER_SPECTRAL_DIR, "train", "Health", "*.tif"))
    if not files:
        print("Warning: No hyperspectral files found for testing.")
        return

    img = load_hyperspectral_image(files[0])
    assert img.ndim == 3, "Spectral image should be 3D"
    assert img.shape[2] == 125, f"Expected 125 bands, got {img.shape[2]}"

    norm_img = preprocess_hyperspectral(img)
    assert norm_img.max() <= 1.0, "Spectral normalization failed"

    features = extract_hyperspectral_features(img)
    assert len(features) == 250, "Spectral feature extraction failed (expected mean+std for 125 bands)"
    print("Spectral Module: PASS")

def test_pest_module():
    print("Testing Pest Module...")
    import glob
    files = glob.glob(os.path.join(PEST_DATASET_DIR, "train", "*", "*.jpg"))
    if not files:
        print("Warning: No pest files found for testing.")
        return

    img = load_pest_image(files[0])
    assert img is not None, "Pest image loading failed"
    assert img.shape == (64, 64, 3), f"Expected (64, 64, 3), got {img.shape}"

    norm_img = preprocess_pest_image(img)
    assert norm_img.max() <= 1.0, "Pest image normalization failed"

    features = extract_pest_features(norm_img)
    assert len(features) == 64*64*3, "Pest feature extraction failed"
    print("Pest Module: PASS")

if __name__ == "__main__":
    try:
        test_crop_module()
        test_spectral_module()
        test_pest_module()
        print("\nALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
