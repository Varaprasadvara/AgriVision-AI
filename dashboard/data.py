import numpy as np
import pandas as pd
import streamlit as st
import tifffile

from src.config import (
    CROP_DATASET_PATH,
    HYPER_SPECTRAL_DIR,
    PEST_CLASSES_PATH,
    PEST_DATASET_DIR,
)
from src.data_loader import load_hyperspectral_image, load_pest_image


CATEGORICAL_CODE_FIELDS = {"soil_type", "water_source_type", "growth_stage"}


def crop_continuous_columns(data):
    return [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in CATEGORICAL_CODE_FIELDS
    ]


@st.cache_data(show_spinner=False)
def crop_data():
    if not CROP_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Crop CSV was not found: {CROP_DATASET_PATH}")
    return pd.read_csv(CROP_DATASET_PATH)


@st.cache_data(show_spinner=False)
def hyperspectral_files(split="train", category="All"):
    split_dir = HYPER_SPECTRAL_DIR / split
    if not split_dir.is_dir():
        return []
    paths = sorted(split_dir.rglob("*.tif"))
    if category != "All":
        paths = [path for path in paths if path.parent.name == category]
    return [str(path) for path in paths]


@st.cache_data(show_spinner=False)
def read_hyperspectral(path):
    image = np.asarray(load_hyperspectral_image(path))
    if image.ndim == 3 and image.shape[0] == 125 and image.shape[-1] != 125:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF cube, received shape {image.shape}")
    return image


@st.cache_data(show_spinner=False)
def hyperspectral_metadata(path):
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        rows = [
            {"Metadata field": "TIFF axes", "Value": tiff.series[0].axes},
            {"Metadata field": "Series shape", "Value": str(tiff.series[0].shape)},
        ]
        interesting_tags = (
            "ImageWidth",
            "ImageLength",
            "BitsPerSample",
            "SamplesPerPixel",
            "ModelPixelScaleTag",
            "ModelTiepointTag",
            "GeoAsciiParamsTag",
        )
        for name in interesting_tags:
            tag = page.tags.get(name)
            if tag is not None:
                rows.append({"Metadata field": name, "Value": str(tag.value)[:180]})
        return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def pest_classes():
    names = {}
    if PEST_CLASSES_PATH.is_file():
        for line in PEST_CLASSES_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                names[int(parts[0]) - 1] = parts[1].strip()
    return names


@st.cache_data(show_spinner=False)
def pest_class_folders(split="train"):
    split_dir = PEST_DATASET_DIR / split
    if not split_dir.is_dir():
        return []
    return sorted(
        (path for path in split_dir.iterdir() if path.is_dir()),
        key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
    )


@st.cache_data(show_spinner=False)
def pest_class_counts(split="train"):
    names = pest_classes()
    rows = []
    for folder in pest_class_folders(split):
        class_id = folder.name
        image_count = sum(1 for _ in folder.glob("*.jpg"))
        label = names.get(int(class_id), "Unmapped class") if class_id.isdigit() else class_id
        rows.append(
            {
                "Class ID": class_id,
                "Class name": label,
                "Images": image_count,
            }
        )
    return pd.DataFrame(rows, columns=["Class ID", "Class name", "Images"])


@st.cache_data(show_spinner=False)
def pest_image_count(split="train"):
    counts = pest_class_counts(split)
    return int(counts["Images"].sum()) if not counts.empty else 0


@st.cache_data(show_spinner=False)
def pest_images(split="train", class_id="All"):
    split_dir = PEST_DATASET_DIR / split
    if not split_dir.is_dir():
        return []
    if class_id == "All":
        folders = pest_class_folders(split)
        paths = []
        for folder in folders:
            paths.extend(folder.glob("*.jpg"))
        return [str(path) for path in sorted(paths)]
    folder = split_dir / str(class_id)
    if not folder.is_dir():
        return []
    return [str(path) for path in sorted(folder.glob("*.jpg"))]


def image_display(path):
    image = load_pest_image(path)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def available_pest_splits():
    return [
        split
        for split in ("train", "val", "test")
        if (PEST_DATASET_DIR / split).is_dir()
    ]


def data_locations():
    return {
        "Crop CSV": CROP_DATASET_PATH,
        "Hyperspectral TIFFs": HYPER_SPECTRAL_DIR,
        "Pest images": PEST_DATASET_DIR,
    }


def band_statistics(image):
    data = image.astype(np.float64, copy=False)
    return pd.DataFrame(
        {
            "Band index": np.arange(data.shape[-1]),
            "Mean": np.nanmean(data, axis=(0, 1)),
            "Std. deviation": np.nanstd(data, axis=(0, 1)),
            "Minimum": np.nanmin(data, axis=(0, 1)),
            "Maximum": np.nanmax(data, axis=(0, 1)),
        }
    )
