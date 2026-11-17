import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import chart_layout
from dashboard.data import (
    available_pest_splits,
    crop_data,
    data_locations,
    hyperspectral_files,
    pest_image_count,
)
from dashboard.theme import navigate_to, page_intro


def render():
    page_intro(
        "AGRICULTURAL INTELLIGENCE",
        "AgriVision AI",
        "Welcome to your agricultural analytics workspace. Explore real crop, spectral, and pest-image data.",
    )
    crop = None
    try:
        crop = crop_data()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop dataset could not be loaded: {exc}")

    hyper_count = sum(len(hyperspectral_files(split)) for split in ("train", "val", "test"))
    pest_count = sum(pest_image_count(split) for split in available_pest_splits())
    cols = st.columns(3)
    cols[0].metric("Crop records", f"{len(crop):,}" if crop is not None else "Unavailable")
    cols[1].metric("Hyperspectral TIFFs", f"{hyper_count:,}")
    cols[2].metric("Pest images", f"{pest_count:,}")

    st.subheader("Workspace at a glance")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        if crop is not None and "label" in crop.columns:
            counts = crop["label"].value_counts().rename_axis("Crop").reset_index(name="Records")
            fig = px.bar(counts, x="Crop", y="Records", title="Crop records by label", color_discrete_sequence=["#16835d"])
            st.plotly_chart(chart_layout(fig, 400), width="stretch")
        else:
            st.info("The crop label chart is unavailable until the crop CSV is accessible.")
    with c2:
        st.markdown("#### Data sources")
        for name, path in data_locations().items():
            available = path.is_file() if name == "Crop CSV" else path.is_dir()
            status = "Available" if available else "Not found"
            st.markdown(
                f'<div class="av-card"><strong>{name}</strong><br>'
                f'<span class="av-muted">{path}</span><br>{status}</div>',
                unsafe_allow_html=True,
            )
            st.write("")

    st.subheader("Analysis toolkit")
    st.caption("Available unsupervised methods")
    st.write("PCA · K-Means · DBSCAN · Isolation Forest")
    activity = st.session_state.get("analysis_history", [])
    if activity:
        st.subheader("Recent analysis activity")
        st.dataframe(pd.DataFrame(activity[-5:][::-1]), width="stretch", hide_index=True)
    else:
        st.caption("Recent activity will appear here after an analysis is run.")

    st.subheader("Quick access")
    quick_cols = st.columns(3)
    quick_pages = ("Dataset Explorer", "Crop & Soil Analysis", "Unsupervised ML Lab")
    for column, target in zip(quick_cols, quick_pages):
        column.button(
            target,
            key=f"overview_{target}",
            on_click=navigate_to,
            args=(target,),
            width="stretch",
        )

    st.caption(
        "Image collections and clusters are explored as supplied data. This dashboard does not "
        "make supervised crop, pest-species, or disease predictions."
    )
