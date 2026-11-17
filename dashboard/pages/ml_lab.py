import streamlit as st

from dashboard.analysis import ALGORITHM_NOTES, run_analysis, save_analysis
from dashboard.components import scatter_plot
from dashboard.data import (
    crop_data,
    hyperspectral_files,
    pest_class_folders,
    pest_classes,
    pest_images,
)
from dashboard.features import (
    crop_matrix,
    feature_summary,
    hyperspectral_matrix,
    pest_matrix,
)
from dashboard.theme import page_intro


def _show_results(results, run_info=None):
    if run_info:
        st.caption(
            f"Latest run · {run_info['Dataset']} · {run_info['Algorithm']} · "
            f"{run_info['Samples']} samples · {run_info['Completed']}"
        )
    chart = scatter_plot(
        results,
        "Component 1",
        "Component 2",
        "Result",
        "Two-component PCA view of the analysis results",
    )
    st.plotly_chart(chart, width="stretch")
    counts = results["Result"].value_counts()
    cols = st.columns(min(4, max(1, len(counts))))
    for index, (label, count) in enumerate(counts.items()):
        cols[index % len(cols)].metric(str(label), f"{count:,}")
    st.dataframe(results, width="stretch", height=320)
    st.download_button(
        "Download analysis results",
        results.to_csv(index=False).encode("utf-8"),
        file_name="agrivision_unsupervised_results.csv",
        mime="text/csv",
    )


def render():
    page_intro(
        "UNSUPERVISED METHODS",
        "Unsupervised ML Lab",
        "Choose a real project dataset, review its feature set, configure an implemented method, and run an analysis.",
    )
    dataset = st.selectbox(
        "Dataset",
        ["Crop & soil records", "Hyperspectral samples", "Pest images"],
    )
    split = "train"
    class_id = None
    feature_mode = "All fields"

    if dataset == "Crop & soil records":
        available = len(crop_data())
        if available < 2:
            st.warning("At least two crop records are required.")
            return
        feature_mode = st.selectbox(
            "Crop feature set",
            ["All fields", "Numeric fields only"],
            help="All fields one-hot encodes categorical inputs; numeric-only excludes text fields.",
        )
        sample_count = st.slider(
            "Records to analyze",
            min_value=2,
            max_value=min(1200, available),
            value=min(300, available),
        )
    elif dataset == "Hyperspectral samples":
        split = st.selectbox("Hyperspectral split", ["train", "val"])
        available = len(hyperspectral_files(split))
        if available < 2:
            st.warning(f"At least two TIFFs are required in the {split} split.")
            return
        sample_count = st.slider(
            "TIFF samples to analyze",
            min_value=2,
            max_value=min(100, available),
            value=min(30, available),
        )
    else:
        splits = [name for name in ("train", "val", "test") if pest_class_folders(name)]
        if not splits:
            st.warning("No pest image folders are available.")
            return
        split = st.selectbox("Pest split", splits)
        folders = pest_class_folders(split)
        ids = [folder.name for folder in folders]
        class_id = st.selectbox(
            "Pest class folder",
            ids,
            format_func=lambda value: (
                f"{value} · {pest_classes().get(int(value), 'Unmapped class')}"
                if value.isdigit()
                else value
            ),
        )
        available = len(pest_images(split, class_id))
        if available < 2:
            st.warning("At least two images are required in the selected class.")
            return
        sample_count = st.slider(
            "Images to analyze",
            min_value=2,
            max_value=min(100, available),
            value=min(30, available),
        )

    algorithm = st.selectbox(
        "Algorithm",
        ["PCA", "K-Means", "DBSCAN", "Isolation Forest"],
    )
    st.info(ALGORITHM_NOTES[algorithm])
    parameters = {}
    if algorithm == "K-Means":
        parameters["clusters"] = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=min(10, sample_count),
            value=min(4, sample_count),
        )
    elif algorithm == "DBSCAN":
        parameters["eps"] = st.slider("DBSCAN epsilon", 0.1, 10.0, 3.0, 0.1)
        parameters["min_samples"] = st.slider(
            "Minimum samples",
            min_value=2,
            max_value=min(20, sample_count),
            value=min(5, sample_count),
        )
    elif algorithm == "Isolation Forest":
        parameters["contamination"] = st.slider(
            "Expected outlier fraction", 0.01, 0.30, 0.10, 0.01
        )

    run_clicked = st.button("Run analysis", type="primary")
    if not run_clicked:
        previous = st.session_state.get("latest_ml_results")
        if previous is not None:
            _show_results(previous, st.session_state.get("latest_ml_run"))
        else:
            st.info("Choose the feature source and settings, then run an analysis.")
        return

    try:
        if dataset == "Crop & soil records":
            matrix, metadata, feature_names = crop_matrix(sample_count, feature_mode)
        elif dataset == "Hyperspectral samples":
            with st.spinner("Reading TIFF cubes and extracting per-band statistics…"):
                matrix, metadata, feature_names = hyperspectral_matrix(split, sample_count)
        else:
            with st.spinner("Reading selected images and extracting pixel features…"):
                matrix, metadata, feature_names = pest_matrix(split, class_id, sample_count)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not prepare features: {exc}")
        return

    st.caption(feature_summary(dataset, feature_names))
    try:
        projection, labels = run_analysis(matrix, algorithm, parameters)
    except (ValueError, FloatingPointError) as exc:
        st.error(f"The selected algorithm could not run on these features: {exc}")
        return

    results = metadata.copy()
    results["Component 1"] = projection[:, 0]
    results["Component 2"] = projection[:, 1]
    results["Result"] = labels
    results["Algorithm"] = algorithm
    results["Dataset"] = dataset
    run_info = save_analysis(dataset, algorithm, len(results), results)
    st.success("Analysis completed using the selected dataset features.")
    _show_results(results, run_info)
    st.warning(
        "Clusters and outliers describe feature similarity only. They do not identify pest species, "
        "crop diseases, infestations, or causal conditions."
    )
