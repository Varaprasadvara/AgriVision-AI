import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import chart_layout, pseudo_rgb
from dashboard.data import (
    band_statistics,
    hyperspectral_files,
    hyperspectral_metadata,
    read_hyperspectral,
)
from dashboard.theme import page_intro


def render():
    page_intro(
        "DATA MANAGEMENT",
        "Hyperspectral Dataset",
        "Browse the original TIFF cubes, inspect their available metadata, and preview actual band values without assuming wavelengths.",
    )
    split = st.selectbox("Dataset split", ["train", "val"], help="The project contains training and validation TIFF cubes.")
    paths = hyperspectral_files(split)
    if not paths:
        st.warning(f"No TIFF cubes were found for the {split} split.")
        return
    categories = sorted({os.path.basename(os.path.dirname(path)) for path in paths})
    category = st.selectbox("Class folder", ["All"] + categories)
    shown = [path for path in paths if category == "All" or os.path.basename(os.path.dirname(path)) == category]
    selected = st.selectbox(
        f"TIFF file · {len(shown):,} available",
        shown,
        format_func=lambda path: os.path.relpath(path),
    )
    try:
        cube = read_hyperspectral(selected)
    except (OSError, ValueError) as exc:
        st.error(f"TIFF cube could not be read: {exc}")
        return

    if cube.shape[-1] != 125:
        st.warning(f"This cube has {cube.shape[-1]} bands; the bundled training sample has 125.")
    if cube.shape[-1] < 3:
        st.error("At least three bands are required to create a color composite.")
        return

    st.caption(f"Actual cube dimensions: {cube.shape[0]} × {cube.shape[1]} × {cube.shape[2]} · {cube.dtype}")
    with st.expander("TIFF metadata"):
        try:
            st.dataframe(hyperspectral_metadata(selected), width="stretch", hide_index=True)
        except (OSError, ValueError) as exc:
            st.error(f"Could not read TIFF metadata: {exc}")
    image_col, spectrum_col = st.columns([1, 1])
    with image_col:
        max_band = cube.shape[-1] - 1
        defaults = [min(40, max_band), min(20, max_band), min(10, max_band)]
        band_cols = st.columns(3)
        red = band_cols[0].number_input("Red display band", 0, max_band, defaults[0], key="rgb_red")
        green = band_cols[1].number_input("Green display band", 0, max_band, defaults[1], key="rgb_green")
        blue = band_cols[2].number_input("Blue display band", 0, max_band, defaults[2], key="rgb_blue")
        st.image(
            pseudo_rgb(cube, int(red), int(green), int(blue)),
            caption=f"Pseudo-RGB display · band indices R {red}, G {green}, B {blue}",
            width="stretch",
        )
        st.caption("Display-only band indices; the TIFFs do not provide verified wavelength mappings.")
    with spectrum_col:
        row = st.slider("Pixel row", 0, cube.shape[0] - 1, cube.shape[0] // 2)
        column = st.slider("Pixel column", 0, cube.shape[1] - 1, cube.shape[1] // 2)
        spectrum = cube[int(row), int(column), :]
        fig = px.line(
            x=np.arange(len(spectrum)),
            y=spectrum,
            labels={"x": "Band index", "y": "Recorded intensity"},
            title="Selected pixel spectrum",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")

    statistics = band_statistics(cube)
    st.subheader("Per-band image statistics")
    stat_col, chart_col = st.columns([1, 1.3])
    with stat_col:
        st.dataframe(statistics, width="stretch", height=380)
        st.download_button(
            "Download band statistics",
            statistics.to_csv(index=False).encode("utf-8"),
            file_name=f"{os.path.splitext(os.path.basename(selected))[0]}_band_statistics.csv",
            mime="text/csv",
        )
    with chart_col:
        fig = px.line(statistics, x="Band index", y=["Mean", "Std. deviation"], title="Mean and variability by band")
        st.plotly_chart(chart_layout(fig, 420), width="stretch")
    with open(selected, "rb") as tiff_file:
        tiff_bytes = tiff_file.read()
    st.download_button(
        "Download original TIFF cube",
        tiff_bytes,
        file_name=os.path.basename(selected),
        mime="image/tiff",
    )
    st.info("NDVI is intentionally not shown: verified red and near-infrared wavelength metadata is unavailable.")
