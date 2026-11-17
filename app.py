import streamlit as st

from dashboard.pages import (
    analytics,
    crop_dataset,
    crop_soil,
    dataset_explorer,
    hyperspectral,
    hyperspectral_analysis,
    ml_lab,
    overview,
    pest_analysis,
    pest_images,
    reports,
)
from dashboard.theme import apply_theme, render_sidebar


st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

page = render_sidebar()
page_renderers = {
    "Dashboard Overview": overview.render,
    "Dataset Explorer": dataset_explorer.render,
    "Crop Dataset": crop_dataset.render,
    "Hyperspectral Dataset": hyperspectral.render,
    "Pest Image Dataset": pest_images.render,
    "Crop & Soil Analysis": crop_soil.render,
    "Hyperspectral Analysis": hyperspectral_analysis.render,
    "Pest Image Analysis": pest_analysis.render,
    "Unsupervised ML Lab": ml_lab.render,
    "Analytics & Visualizations": analytics.render,
    "Reports & Downloads": reports.render,
}

page_renderers[page]()
