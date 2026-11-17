import streamlit as st

from dashboard.analysis import ALGORITHM_NOTES, run_analysis, save_analysis
from dashboard.components import scatter_plot
from dashboard.data import hyperspectral_files
from dashboard.features import feature_summary, hyperspectral_matrix
from dashboard.theme import page_intro


def render():
    page_intro(
        "SPECTRAL ANALYSIS",
        "Hyperspectral Analysis",
        "Analyze actual TIFF samples using per-band image means and standard deviations.",
    )
    split = st.selectbox("Dataset split", ["train", "val"], key="hyper_analysis_split")
    available = len(hyperspectral_files(split))
    if available < 2:
        st.warning(f"At least two TIFF cubes are required in the {split} split.")
        return
    sample_count = st.slider(
        "TIFF samples",
        min_value=2,
        max_value=min(100, available),
        value=min(30, available),
    )
    algorithm = st.selectbox(
        "Algorithm",
        ["PCA", "K-Means", "DBSCAN", "Isolation Forest"],
        key="hyper_analysis_algorithm",
    )
    parameters = {}
    if algorithm == "K-Means":
        parameters["clusters"] = st.slider(
            "Number of clusters", 2, min(10, sample_count), min(4, sample_count)
        )
    elif algorithm == "DBSCAN":
        parameters["eps"] = st.slider("DBSCAN epsilon", 0.1, 10.0, 3.0, 0.1)
        parameters["min_samples"] = st.slider(
            "Minimum samples", 2, min(20, sample_count), min(5, sample_count)
        )
    elif algorithm == "Isolation Forest":
        parameters["contamination"] = st.slider(
            "Expected outlier fraction", 0.01, 0.30, 0.10, 0.01
        )
    st.info(ALGORITHM_NOTES[algorithm])

    if st.button("Run spectral analysis", type="primary"):
        try:
            with st.spinner("Reading actual TIFF cubes and computing per-band features…"):
                matrix, metadata, feature_names = hyperspectral_matrix(split, sample_count)
            projection, labels = run_analysis(matrix, algorithm, parameters)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Spectral analysis could not run: {exc}")
            return
        results = metadata.copy()
        results["Component 1"] = projection[:, 0]
        results["Component 2"] = projection[:, 1]
        results["Result"] = labels
        results["Algorithm"] = algorithm
        results["Dataset"] = "Hyperspectral samples"
        run_info = save_analysis("Hyperspectral samples", algorithm, len(results), results)
        st.success(f"{algorithm} completed on {len(results):,} TIFF samples.")
        st.caption(feature_summary("Hyperspectral samples", feature_names))
        st.caption(
            f"Run: {run_info['Completed']} · {split} split · "
            "125-band mean and standard-deviation feature vectors"
        )
        fig = scatter_plot(
            results,
            "Component 1",
            "Component 2",
            "Result",
            f"{algorithm} results shown in PCA space",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(results, width="stretch", height=320)
        st.download_button(
            "Download spectral analysis",
            results.to_csv(index=False).encode("utf-8"),
            file_name="agrivision_hyperspectral_analysis.csv",
            mime="text/csv",
        )
        st.warning(
            "Results summarize spectral feature similarity only. No wavelengths, NDVI, crop disease, "
            "or causal condition are inferred."
        )
