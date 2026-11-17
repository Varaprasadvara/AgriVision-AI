from datetime import datetime

import numpy as np
import streamlit as st

from src.unsupervised_learning import run_anomaly_detection, run_dbscan, run_kmeans, run_pca


ALGORITHM_NOTES = {
    "PCA": "Projects standardized feature vectors onto two principal components; no clusters are assigned.",
    "K-Means": "Partitions standardized feature vectors into the selected number of similarity groups.",
    "DBSCAN": "Groups dense neighborhoods and marks sparse points as noise; results depend on epsilon and scale.",
    "Isolation Forest": "Flags feature vectors that are easier to isolate; an outlier is not a diagnosis.",
}


def run_analysis(matrix, algorithm, parameters):
    projection = run_pca(matrix)
    if algorithm == "PCA":
        labels = np.repeat("Projection only", len(matrix))
    elif algorithm == "K-Means":
        labels = run_kmeans(matrix, n_clusters=parameters["clusters"]).astype(str)
    elif algorithm == "DBSCAN":
        labels = run_dbscan(
            matrix,
            eps=parameters["eps"],
            min_samples=parameters["min_samples"],
        ).astype(str)
    elif algorithm == "Isolation Forest":
        raw_labels = run_anomaly_detection(
            matrix,
            contamination=parameters["contamination"],
        )
        labels = np.where(raw_labels == -1, "Outlier", "Inlier")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return projection, labels


def save_analysis(dataset, algorithm, sample_count, results):
    record = {
        "Completed": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "Dataset": dataset,
        "Algorithm": algorithm,
        "Samples": int(sample_count),
        "Output groups": int(results["Result"].nunique()),
    }
    history = list(st.session_state.get("analysis_history", []))
    history.append(record)
    st.session_state["analysis_history"] = history[-50:]
    st.session_state["latest_ml_run"] = record
    st.session_state["latest_ml_results"] = results.copy()
    return record
