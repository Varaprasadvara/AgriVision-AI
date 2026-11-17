import os

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import chart_layout
from dashboard.data import (
    available_pest_splits,
    band_statistics,
    crop_data,
    data_locations,
    hyperspectral_files,
    pest_class_counts,
    read_hyperspectral,
)
from dashboard.theme import page_intro
from src.config import CROP_DATASET_PATH


def _inventory_report():
    rows = []
    try:
        rows.append(
            {
                "Dataset": "Crop & soil",
                "Split": "all",
                "Records": len(crop_data()),
                "Location": str(data_locations()["Crop CSV"]),
            }
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be loaded: {exc}")
    for split in ("train", "val", "test"):
        count = len(hyperspectral_files(split))
        if count:
            rows.append(
                {
                    "Dataset": "Hyperspectral TIFF",
                    "Split": split,
                    "Records": count,
                    "Location": str(data_locations()["Hyperspectral TIFFs"]),
                }
            )
    for split in available_pest_splits():
        counts = pest_class_counts(split)
        rows.append(
            {
                "Dataset": "Pest images",
                "Split": split,
                "Records": int(counts["Images"].sum()),
                "Location": str(data_locations()["Pest images"]),
            }
        )
    return pd.DataFrame(rows, columns=["Dataset", "Split", "Records", "Location"])


def _crop_summary():
    try:
        data = crop_data()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be loaded: {exc}")
        return
    if "label" not in data:
        st.warning("The crop CSV does not contain a label field for a crop summary.")
        return
    aggregations = {"Records": ("label", "size")}
    for column in ("N", "P", "K", "ph", "rainfall"):
        if column in data:
            aggregations[f"Mean_{column}"] = (column, "mean")
    summary = data.groupby("label", dropna=False).agg(**aggregations).reset_index()
    st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        "Download crop-label summary",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="agrivision_crop_label_summary.csv",
        mime="text/csv",
    )


def _band_report():
    split = st.selectbox("TIFF split", ["train", "val"], key="report_hyper_split")
    files = hyperspectral_files(split)
    if not files:
        st.warning(f"No TIFF files are available in {split}.")
        return
    selected = st.selectbox("TIFF sample", files, format_func=os.path.basename)
    try:
        stats = band_statistics(read_hyperspectral(selected))
    except (OSError, ValueError) as exc:
        st.error(f"Could not generate band statistics: {exc}")
        return
    st.dataframe(stats, width="stretch", height=330)
    st.download_button(
        "Download band-statistics report",
        stats.to_csv(index=False).encode("utf-8"),
        file_name=f"{os.path.splitext(os.path.basename(selected))[0]}_band_report.csv",
        mime="text/csv",
    )


def render():
    page_intro(
        "REPORTING",
        "Reports & Downloads",
        "Export dataset summaries and completed analysis results. Reports are generated from the project’s actual files and session results.",
    )
    report_type = st.selectbox(
        "Report",
        [
            "Dataset inventory",
            "Crop-label summary",
            "Hyperspectral band statistics",
            "Pest class counts",
            "Latest ML execution",
        ],
    )

    if report_type == "Dataset inventory":
        report = _inventory_report()
        if report.empty:
            st.warning("No datasets were found to include in the inventory.")
            return
        st.dataframe(report, width="stretch", hide_index=True)
        st.download_button(
            "Download dataset inventory",
            report.to_csv(index=False).encode("utf-8"),
            file_name="agrivision_dataset_inventory.csv",
            mime="text/csv",
        )
        totals = report.groupby("Dataset", as_index=False)["Records"].sum()
        fig = px.bar(totals, x="Dataset", y="Records", title="Dataset record counts", color_discrete_sequence=["#16835d"])
        st.plotly_chart(chart_layout(fig), width="stretch")
        st.download_button(
            "Download chart as interactive HTML",
            fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8"),
            file_name="agrivision_dataset_inventory_chart.html",
            mime="text/html",
        )
    elif report_type == "Crop-label summary":
        _crop_summary()
        if CROP_DATASET_PATH.is_file():
            st.download_button(
                "Download original crop CSV",
                CROP_DATASET_PATH.read_bytes(),
                file_name=CROP_DATASET_PATH.name,
                mime="text/csv",
            )
    elif report_type == "Hyperspectral band statistics":
        _band_report()
    elif report_type == "Pest class counts":
        splits = available_pest_splits()
        if not splits:
            st.warning("No pest dataset splits are available.")
            return
        split = st.selectbox("Image split", splits, key="report_pest_split")
        counts = pest_class_counts(split)
        st.dataframe(counts, width="stretch", hide_index=True)
        st.download_button(
            "Download pest class counts",
            counts.to_csv(index=False).encode("utf-8"),
            file_name=f"pest_{split}_class_counts.csv",
            mime="text/csv",
        )
    else:
        results = st.session_state.get("latest_ml_results")
        run = st.session_state.get("latest_ml_run")
        history = pd.DataFrame(st.session_state.get("analysis_history", []))
        if not history.empty:
            st.subheader("Execution history")
            st.dataframe(history, width="stretch", hide_index=True)
            st.download_button(
                "Download execution history",
                history.to_csv(index=False).encode("utf-8"),
                file_name="agrivision_analysis_history.csv",
                mime="text/csv",
            )
        if results is None or run is None:
            st.info("No ML analysis has been run in this session yet.")
            return
        st.subheader("Latest completed execution")
        st.json(run)
        st.dataframe(results, width="stretch", height=320)
        st.download_button(
            "Download latest ML results",
            results.to_csv(index=False).encode("utf-8"),
            file_name="agrivision_latest_ml_results.csv",
            mime="text/csv",
        )
        st.caption("PDF export is not enabled; CSV and interactive HTML are available without adding a PDF dependency.")
