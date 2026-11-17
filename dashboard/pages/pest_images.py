import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from dashboard.components import chart_layout
from dashboard.data import (
    available_pest_splits,
    pest_class_counts,
    pest_class_folders,
    pest_classes,
    pest_images,
)
from dashboard.theme import page_intro


def render():
    page_intro(
        "DATA MANAGEMENT",
        "Pest Image Dataset",
        "Browse the supplied labeled image splits and classes; only the selected class images are loaded for preview.",
    )
    splits = available_pest_splits()
    if not splits:
        st.warning("No pest image split folders were found. Check the configured pest dataset path.")
        return
    split = st.selectbox("Dataset split", splits)
    class_counts = pest_class_counts(split)
    folders = pest_class_folders(split)
    if class_counts.empty or not folders:
        st.warning(f"No class folders were found in the {split} split.")
        return
    names = pest_classes()
    options = [folder.name for folder in folders]

    def format_class(class_id):
        if class_id.isdigit():
            label = names.get(int(class_id))
            return f"{class_id} · {label}" if label else f"{class_id} · Unmapped class"
        return class_id

    class_id = st.selectbox("Class folder", options, format_func=format_class)
    class_row = class_counts[class_counts["Class ID"] == class_id]
    count = int(class_row["Images"].iloc[0]) if not class_row.empty else 0
    image_paths = pest_images(split, class_id)
    metrics = st.columns(3)
    metrics[0].metric("Images in split", f"{int(class_counts['Images'].sum()):,}")
    metrics[1].metric("Class folders", f"{len(class_counts):,}")
    metrics[2].metric("Selected class images", f"{count:,}")

    st.subheader("Class distribution")
    top_counts = class_counts.sort_values("Images", ascending=False).head(20)
    fig = px.bar(
        top_counts,
        x="Class ID",
        y="Images",
        hover_name="Class name",
        title="Largest 20 classes in this split",
        color_discrete_sequence=["#16835d"],
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    with st.expander("All class counts"):
        st.dataframe(class_counts, width="stretch", hide_index=True)
        st.download_button(
            "Download class counts",
            class_counts.to_csv(index=False).encode("utf-8"),
            file_name=f"pest_{split}_class_counts.csv",
            mime="text/csv",
        )

    if not image_paths:
        st.info("No JPEG images were found in the selected class folder.")
        return
    selected_image = st.selectbox("Inspect an image", image_paths, format_func=os.path.basename)
    image_column, feature_column = st.columns([1, 1])
    try:
        with Image.open(selected_image) as image:
            rgb_image = image.convert("RGB")
            rgb_image.load()
            pixels = np.asarray(rgb_image)
            image_column.image(rgb_image, caption=os.path.basename(selected_image), width="stretch")
            image_column.caption(
                f"Original image dimensions: {rgb_image.width} × {rgb_image.height} pixels"
            )
        channel_means = pd.DataFrame(
            {"Channel": ["Red", "Green", "Blue"], "Mean pixel value": pixels.mean(axis=(0, 1))}
        )
        feature_column.markdown("#### Basic image features")
        feature_column.caption("Mean RGB intensity from the selected original image.")
        feature_column.dataframe(channel_means, width="stretch", hide_index=True)
        with open(selected_image, "rb") as image_file:
            image_bytes = image_file.read()
        st.download_button(
            "Download selected original image",
            image_bytes,
            file_name=os.path.basename(selected_image),
            mime="image/jpeg",
        )
    except (OSError, ValueError) as exc:
        st.error(f"Could not inspect the selected image: {exc}")

    st.subheader("Image gallery")
    page_size = 8
    max_start = max(0, len(image_paths) - page_size)
    start = st.number_input("Start at image", min_value=0, max_value=max_start, value=0, step=page_size)
    columns = st.columns(4)
    for index, path in enumerate(image_paths[int(start):int(start) + page_size]):
        try:
            with Image.open(path) as image:
                image.load()
                columns[index % 4].image(
                    image.copy(),
                    caption=os.path.basename(path),
                    width="stretch",
                )
        except (OSError, ValueError) as exc:
            columns[index % 4].error(f"{os.path.basename(path)} could not be opened: {exc}")

    st.caption("Image folders and names follow the supplied dataset labels; these previews are not model predictions.")
