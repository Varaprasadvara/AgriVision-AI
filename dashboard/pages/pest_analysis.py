import os

import streamlit as st
from PIL import Image

from dashboard.analysis import ALGORITHM_NOTES, run_analysis, save_analysis
from dashboard.components import scatter_plot
from dashboard.data import (
    pest_class_folders,
    pest_classes,
    pest_images,
)
from dashboard.features import feature_summary, pest_matrix
from dashboard.theme import page_intro


def render():
    page_intro(
        "IMAGE FEATURE ANALYSIS",
        "Pest Image Analysis",
        "Explore pixel-feature similarity within a selected image class; unsupervised groups do not confirm pest identity or infestation.",
    )
    split = st.selectbox("Dataset split", ["train", "val", "test"], key="pest_analysis_split")
    folders = pest_class_folders(split)
    if not folders:
        st.warning(f"No labeled image folders were found in {split}.")
        return
    names = pest_classes()
    class_ids = [folder.name for folder in folders]
    class_id = st.selectbox(
        "Class folder",
        class_ids,
        format_func=lambda value: (
            f"{value} · {names.get(int(value), 'Unmapped class')}"
            if value.isdigit()
            else value
        ),
    )
    paths = pest_images(split, class_id)
    if len(paths) < 2:
        st.warning("At least two images are required to compare image features.")
        return
    sample_count = st.slider(
        "Images to compare",
        min_value=2,
        max_value=min(100, len(paths)),
        value=min(30, len(paths)),
    )
    algorithm = st.selectbox(
        "Algorithm",
        ["K-Means", "DBSCAN", "PCA", "Isolation Forest"],
        key="pest_analysis_algorithm",
    )
    parameters = {}
    if algorithm == "K-Means":
        parameters["clusters"] = st.slider(
            "Number of clusters", 2, min(10, sample_count), min(4, sample_count)
        )
    elif algorithm == "DBSCAN":
        parameters["eps"] = st.slider("DBSCAN epsilon", 0.1, 15.0, 3.0, 0.1)
        parameters["min_samples"] = st.slider(
            "Minimum samples", 2, min(20, sample_count), min(5, sample_count)
        )
    elif algorithm == "Isolation Forest":
        parameters["contamination"] = st.slider(
            "Expected outlier fraction", 0.01, 0.30, 0.10, 0.01
        )
    st.info(ALGORITHM_NOTES[algorithm])

    if st.button("Run image analysis", type="primary"):
        try:
            with st.spinner("Extracting RGB pixel features from selected real images…"):
                matrix, metadata, feature_names = pest_matrix(split, class_id, sample_count)
            projection, labels = run_analysis(matrix, algorithm, parameters)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Image analysis could not run: {exc}")
            return
        results = metadata.copy()
        results["Component 1"] = projection[:, 0]
        results["Component 2"] = projection[:, 1]
        results["Result"] = labels
        results["Algorithm"] = algorithm
        results["Dataset"] = "Pest images"
        run_info = save_analysis("Pest images", algorithm, len(results), results)
        st.success(f"{algorithm} completed on {len(results):,} actual images.")
        st.caption(feature_summary("Pest images", feature_names))
        st.caption(
            f"Run: {run_info['Completed']} · class folder {class_id} · split {split}"
        )
        fig = scatter_plot(
            results,
            "Component 1",
            "Component 2",
            "Result",
            f"{algorithm} results shown in PCA space",
            hover_data=["Record"],
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(results, width="stretch", height=320)
        st.download_button(
            "Download image analysis",
            results.to_csv(index=False).encode("utf-8"),
            file_name="agrivision_pest_image_analysis.csv",
            mime="text/csv",
        )

        st.subheader("Representative images")
        path_by_name = {os.path.basename(path): path for path in paths}
        representatives = results.drop_duplicates("Result").head(4)
        columns = st.columns(max(1, len(representatives)))
        for index, row in enumerate(representatives.itertuples(index=False)):
            path = path_by_name.get(row.Record)
            if path is None:
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    columns[index].image(
                        image.copy(),
                        caption=f"{row.Result} · {row.Record}",
                        width="stretch",
                    )
            except (OSError, ValueError) as exc:
                columns[index].error(f"{row.Record} could not be opened: {exc}")
        st.warning(
            "Clusters describe visual feature similarity within this selected folder; "
            "they do not confirm species, disease, or infestation."
        )
