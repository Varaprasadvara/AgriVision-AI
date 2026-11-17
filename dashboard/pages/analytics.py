import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from dashboard.components import chart_layout
from dashboard.data import (
    available_pest_splits,
    band_statistics,
    crop_continuous_columns,
    crop_data,
    hyperspectral_files,
    pest_class_counts,
    pest_class_folders,
    pest_images,
    read_hyperspectral,
)
from dashboard.theme import page_intro


def _show_latest_ml():
    results = st.session_state.get("latest_ml_results")
    if results is None or results.empty:
        st.info("Run an analysis in the Unsupervised ML Lab to add a PCA, cluster, or anomaly chart here.")
        return
    run = st.session_state.get("latest_ml_run", {})
    st.caption(
        f"{run.get('Dataset', 'Dataset')} · {run.get('Algorithm', 'Algorithm')} · "
        f"{run.get('Samples', len(results))} samples · {run.get('Completed', '')}"
    )
    fig = px.scatter(
        results,
        x="Component 1",
        y="Component 2",
        color="Result",
        hover_name="Record",
        title="Latest saved unsupervised result",
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.dataframe(results, width="stretch", height=280)


def _crop_analytics(analysis_type):
    try:
        data = crop_data()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be loaded: {exc}")
        return
    if analysis_type == "Crop label counts":
        if "label" not in data:
            st.warning("The crop CSV does not include a label column.")
            return
        summary = data["label"].value_counts().rename_axis("Crop").reset_index(name="Records")
        fig = px.bar(summary, x="Crop", y="Records", title="Crop records by label", color_discrete_sequence=["#16835d"])
        st.plotly_chart(chart_layout(fig), width="stretch")
    elif analysis_type == "Feature distribution":
        numeric = crop_continuous_columns(data)
        if not numeric:
            st.warning("No numeric crop fields are available.")
            return
        feature = st.selectbox("Feature", numeric)
        fig = px.histogram(
            data,
            x=feature,
            color="label" if "label" in data else None,
            marginal="box",
            title=f"Observed distribution · {feature}",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    else:
        numeric = data[crop_continuous_columns(data)]
        correlation = numeric.corr()
        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Crop and environmental feature correlation",
        )
        st.plotly_chart(chart_layout(fig, 620), width="stretch")
    st.download_button(
        "Download crop data used",
        data.to_csv(index=False).encode("utf-8"),
        file_name="agrivision_crop_analytics_source.csv",
        mime="text/csv",
    )


def _hyperspectral_analytics(analysis_type):
    split = st.selectbox("TIFF split", ["train", "val"], key="analytics_hyper_split")
    files = hyperspectral_files(split)
    if not files:
        st.warning(f"No TIFF cubes were found for the {split} split.")
        return
    selected = st.selectbox("TIFF image", files, format_func=os.path.basename)
    try:
        cube = read_hyperspectral(selected)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read the selected TIFF: {exc}")
        return
    stats = band_statistics(cube)
    metric = {
        "Band mean profile": "Mean",
        "Band variability": "Std. deviation",
        "Band range": "Maximum",
    }[analysis_type]
    fig = px.line(
        stats,
        x="Band index",
        y=metric,
        title=f"{metric} by band · {os.path.basename(selected)}",
    )
    st.caption(f"Image shape: {cube.shape} · dtype: {cube.dtype}")
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.dataframe(stats, width="stretch", height=280)
    st.download_button(
        "Download band profile",
        stats.to_csv(index=False).encode("utf-8"),
        file_name=f"{os.path.splitext(os.path.basename(selected))[0]}_band_report.csv",
        mime="text/csv",
    )


def _pest_analytics(analysis_type):
    splits = available_pest_splits()
    if not splits:
        st.warning("No pest dataset splits were found.")
        return
    if analysis_type == "Images by split":
        counts = pd.DataFrame(
            {
                "Split": splits,
                "Images": [int(pest_class_counts(split)["Images"].sum()) for split in splits],
            }
        )
        fig = px.bar(counts, x="Split", y="Images", title="Pest images by split", color_discrete_sequence=["#16835d"])
        st.plotly_chart(chart_layout(fig), width="stretch")
        st.dataframe(counts, width="stretch", hide_index=True)
        return
    split = st.selectbox("Pest split", splits, key="analytics_pest_split")
    counts = pest_class_counts(split)
    if counts.empty:
        st.info(f"No class folders were found in {split}.")
        return
    if analysis_type == "Class image counts":
        top = counts.sort_values("Images", ascending=False).head(30)
        fig = px.bar(
            top,
            x="Class ID",
            y="Images",
            hover_name="Class name",
            title=f"Image counts by class · {split} · largest 30",
            color_discrete_sequence=["#16835d"],
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
        st.dataframe(counts, width="stretch", hide_index=True)
        st.download_button(
            "Download class counts",
            counts.to_csv(index=False).encode("utf-8"),
            file_name=f"pest_{split}_class_counts.csv",
            mime="text/csv",
        )
        return
    folders = pest_class_folders(split)
    class_id = st.selectbox("Class folder", [folder.name for folder in folders], key="analytics_pest_class")
    files = pest_images(split, class_id)
    if not files:
        st.info("No images are available for this class.")
        return
    with Image.open(files[0]) as image:
        image.load()
        st.image(image.copy(), caption=os.path.basename(files[0]), width="stretch")
        st.caption(f"Image feature preview · original dimensions {image.width} × {image.height}")


def _dataset_inventory():
    report_rows = []
    try:
        report_rows.append({"Dataset": "Crop & soil", "Records": len(crop_data())})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be loaded: {exc}")
    hyper_records = sum(len(hyperspectral_files(split)) for split in ("train", "val", "test"))
    pest_records = sum(
        int(pest_class_counts(split)["Images"].sum())
        for split in available_pest_splits()
    )
    report_rows.extend([
        {"Dataset": "Hyperspectral TIFF", "Records": hyper_records},
        {"Dataset": "Pest images", "Records": pest_records},
    ])
    report = pd.DataFrame(report_rows)
    fig = px.bar(report, x="Dataset", y="Records", title="Actual dataset inventory", color_discrete_sequence=["#16835d"])
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.dataframe(report, width="stretch", hide_index=True)


def render():
    page_intro(
        "ANALYTICS",
        "Analytics & Visualizations",
        "Select a dataset and analysis to explore charts built from real records, images, and saved ML runs.",
    )
    datasets = ["All datasets", "Crop & soil", "Hyperspectral", "Pest images"]
    dataset = st.selectbox("Dataset", datasets)
    latest = st.session_state.get("latest_ml_results")
    if dataset == "All datasets":
        analysis_types = ["Dataset inventory"]
        if latest is not None:
            analysis_types.append("Latest ML result")
    elif dataset == "Crop & soil":
        analysis_types = ["Crop label counts", "Feature distribution", "Feature correlation"]
    elif dataset == "Hyperspectral":
        analysis_types = ["Band mean profile", "Band variability", "Band range"]
    else:
        analysis_types = ["Class image counts", "Images by split", "Class image preview"]
    analysis_type = st.selectbox("Analysis view", analysis_types)

    if dataset == "All datasets":
        if analysis_type == "Dataset inventory":
            _dataset_inventory()
        else:
            _show_latest_ml()
    elif dataset == "Crop & soil":
        _crop_analytics(analysis_type)
    elif dataset == "Hyperspectral":
        _hyperspectral_analytics(analysis_type)
    else:
        _pest_analytics(analysis_type)
