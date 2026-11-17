import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from dashboard.data import (
    CATEGORICAL_CODE_FIELDS,
    crop_data,
    crop_continuous_columns,
    hyperspectral_files,
    image_display,
    pest_images,
    read_hyperspectral,
)
from src.feature_extraction import extract_hyperspectral_features, extract_pest_features
from src.preprocessing import preprocess_pest_image


@st.cache_data(show_spinner=False)
def crop_matrix(sample_count, feature_mode="All fields"):
    frame = crop_data()
    sample = frame.sample(n=min(sample_count, len(frame)), random_state=42).reset_index(drop=True)
    metadata = pd.DataFrame({"Record": [f"Crop record {i + 1}" for i in range(len(sample))]})
    if "label" in sample:
        metadata["Crop label"] = sample["label"].astype(str)
    features = sample.drop(columns=["label"], errors="ignore")
    if feature_mode == "Numeric fields only":
        continuous = crop_continuous_columns(features)
        features = features[continuous]
    else:
        coded_categories = [column for column in CATEGORICAL_CODE_FIELDS if column in features]
        for column in coded_categories:
            features[column] = features[column].astype(str)
        features = pd.get_dummies(features, dummy_na=True, dtype=float)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)
    values = features.to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(values), metadata, features.columns.tolist()


@st.cache_data(show_spinner=False)
def hyperspectral_matrix(split, sample_count):
    paths = hyperspectral_files(split)
    if not paths:
        raise ValueError(f"No hyperspectral TIFFs are available in {split}.")
    selection = paths[:min(sample_count, len(paths))]
    raw = np.vstack([
        extract_hyperspectral_features(read_hyperspectral(path))
        for path in selection
    ])
    feature_names = (
        [f"Mean band {i}" for i in range(raw.shape[1] // 2)]
        + [f"Standard deviation band {i}" for i in range(raw.shape[1] // 2)]
    )
    metadata = pd.DataFrame({"Record": [os.path.basename(path) for path in selection]})
    return StandardScaler().fit_transform(raw), metadata, feature_names


@st.cache_data(show_spinner=False)
def pest_matrix(split, class_id, sample_count):
    paths = pest_images(split, class_id)
    if not paths:
        raise ValueError(f"No pest images are available for class {class_id} in {split}.")
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(paths, size=min(sample_count, len(paths)), replace=False).tolist())
    rows = []
    for path in chosen:
        image = image_display(path)
        rows.append(extract_pest_features(preprocess_pest_image(image)))
    raw = np.asarray(rows, dtype=np.float64)
    feature_names = [f"RGB pixel feature {i + 1}" for i in range(raw.shape[1])]
    matrix = StandardScaler().fit_transform(raw)
    return matrix, pd.DataFrame({"Record": [os.path.basename(path) for path in chosen]}), feature_names


def feature_summary(dataset, feature_names):
    if dataset == "Hyperspectral samples":
        return f"{len(feature_names)} features: per-band image means and standard deviations."
    if dataset == "Pest images":
        return f"{len(feature_names):,} features: flattened RGB pixel values from resized images."
    preview = ", ".join(feature_names[:12])
    remainder = len(feature_names) - min(12, len(feature_names))
    suffix = f", and {remainder} more" if remainder else ""
    return f"{len(feature_names)} features: {preview}{suffix}."
