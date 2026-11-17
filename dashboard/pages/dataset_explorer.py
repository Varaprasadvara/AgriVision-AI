import os

import pandas as pd
import streamlit as st
from PIL import Image

from dashboard.components import pseudo_rgb
from dashboard.data import (
    available_pest_splits,
    band_statistics,
    crop_data,
    data_locations,
    hyperspectral_files,
    pest_class_counts,
    pest_class_folders,
    pest_classes,
    pest_image_count,
    pest_images,
    read_hyperspectral,
)
from dashboard.theme import navigate_to, page_intro


DATASET_OPTIONS = ["Crop CSV", "Hyperspectral TIFF", "Pest images"]
DATASET_PAGES = {
    "Crop CSV": "Crop Dataset",
    "Hyperspectral TIFF": "Hyperspectral Dataset",
    "Pest images": "Pest Image Dataset",
}
LOCATION_KEYS = {
    "Crop CSV": "Crop CSV",
    "Hyperspectral TIFF": "Hyperspectral TIFFs",
    "Pest images": "Pest images",
}


def _choose_dataset(dataset):
    st.session_state["dataset_selection"] = dataset


def _dataset_summary():
    crop_count = None
    try:
        crop_count = len(crop_data())
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be inspected: {exc}")
    hyper_count = sum(len(hyperspectral_files(split)) for split in ("train", "val", "test"))
    pest_count = sum(pest_image_count(split) for split in available_pest_splits())
    return {
        "Crop CSV": f"{crop_count:,} records" if crop_count is not None else "Unavailable",
        "Hyperspectral TIFF": f"{hyper_count:,} files",
        "Pest images": f"{pest_count:,} images",
    }


def _preview_crop():
    try:
        data = crop_data()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be loaded: {exc}")
        return
    query = st.text_input("Search crop records", key="explorer_crop_search").strip().casefold()
    filtered = data
    if query:
        matches = data.astype(str).apply(
            lambda column: column.str.casefold().str.contains(query, regex=False, na=False)
        ).any(axis=1)
        filtered = data.loc[matches]
    st.caption(f"{len(filtered):,} matching rows · {len(data.columns)} columns")
    st.dataframe(filtered.head(100), width="stretch", height=330)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download original CSV",
        data.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(str(data_locations()["Crop CSV"])),
        mime="text/csv",
    )
    c2.download_button(
        "Download matching rows",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="agrivision_crop_selection.csv",
        mime="text/csv",
    )


def _preview_hyperspectral():
    split = st.selectbox("TIFF split", ["train", "val"])
    paths = hyperspectral_files(split)
    if not paths:
        st.warning(f"No hyperspectral TIFFs are available in {split}.")
        return
    selected = st.selectbox("TIFF file", paths, format_func=os.path.basename)
    try:
        cube = read_hyperspectral(selected)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read the selected TIFF: {exc}")
        return
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.image(pseudo_rgb(cube), caption="Band-index pseudo-RGB preview", width="stretch")
        st.caption("Display composite only; verified wavelengths are not available.")
    with c2:
        st.metric("Image dimensions", f"{cube.shape[0]} × {cube.shape[1]}")
        st.metric("Spectral bands", cube.shape[-1])
        stats = band_statistics(cube)
        st.dataframe(stats.head(12), width="stretch", hide_index=True)
    st.download_button(
        "Download band statistics",
        stats.to_csv(index=False).encode("utf-8"),
        file_name=f"{os.path.splitext(os.path.basename(selected))[0]}_bands.csv",
        mime="text/csv",
    )


def _preview_pests():
    splits = available_pest_splits()
    if not splits:
        st.warning("No pest dataset splits were found.")
        return
    split = st.selectbox("Image split", splits)
    counts = pest_class_counts(split)
    folders = pest_class_folders(split)
    if counts.empty or not folders:
        st.warning(f"No labeled image folders were found in {split}.")
        return
    class_ids = [folder.name for folder in folders]
    names = pest_classes()
    class_id = st.selectbox(
        "Class folder",
        class_ids,
        format_func=lambda value: (
            f"{value} · {names.get(int(value), 'Unmapped class')}"
            if value.isdigit()
            else value
        ),
    )
    images = pest_images(split, class_id)
    class_count = counts[counts["Class ID"] == class_id]
    st.caption(f"{pest_image_count(split):,} images in {len(counts):,} class folders for this split.")
    st.dataframe(class_count, width="stretch", hide_index=True)
    if not images:
        st.info("The selected class has no JPEG images.")
        return
    image_path = st.selectbox("Image file", images, format_func=os.path.basename)
    try:
        with Image.open(image_path) as image:
            image.load()
            st.image(image.copy(), caption=os.path.basename(image_path), width="stretch")
            st.caption(f"Original image dimensions: {image.width} × {image.height}")
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        st.download_button(
            "Download selected image",
            image_bytes,
            file_name=os.path.basename(image_path),
            mime="image/jpeg",
        )
    except (OSError, ValueError) as exc:
        st.error(f"Could not preview the selected image: {exc}")


def render():
    page_intro(
        "DATA CATALOG",
        "Dataset Explorer",
        "Access each project dataset independently, preview real records or images, and open its focused workspace.",
    )
    summaries = _dataset_summary()
    locations = data_locations()
    cols = st.columns(3)
    for column, dataset in zip(cols, DATASET_OPTIONS):
        path = locations[LOCATION_KEYS[dataset]]
        available = path.is_file() if dataset == "Crop CSV" else path.is_dir()
        column.markdown(
            f'<div class="av-card"><p class="av-eyebrow">{dataset}</p>'
            f'<h3>{summaries[dataset]}</h3><span class="av-muted">{path}</span><br>'
            f'{"Available" if available else "Unavailable"}</div>',
            unsafe_allow_html=True,
        )
        column.button(
            "Preview dataset",
            key=f"preview_{dataset}",
            on_click=_choose_dataset,
            args=(dataset,),
            width="stretch",
            disabled=not available,
        )
        column.button(
            "Open dataset page",
            key=f"explore_{dataset}",
            on_click=navigate_to,
            args=(DATASET_PAGES[dataset],),
            width="stretch",
            disabled=not available,
        )

    st.divider()
    selected = st.selectbox("Preview a dataset", DATASET_OPTIONS, key="dataset_selection")
    if selected == "Crop CSV":
        _preview_crop()
    elif selected == "Hyperspectral TIFF":
        _preview_hyperspectral()
    else:
        _preview_pests()
