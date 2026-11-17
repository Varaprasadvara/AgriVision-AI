import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.analysis import ALGORITHM_NOTES, run_analysis, save_analysis
from dashboard.components import chart_layout
from dashboard.data import crop_continuous_columns, crop_data
from dashboard.features import crop_matrix
from dashboard.theme import page_intro


def render():
    page_intro(
        "FIELD ANALYSIS",
        "Crop & Soil Analysis",
        "Compare observed crop and environmental features; optional clusters summarize numeric feature similarity.",
    )
    try:
        data = crop_data()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop dataset could not be loaded: {exc}")
        return
    numeric = data[crop_continuous_columns(data)]
    if numeric.empty:
        st.error("No numeric crop or environmental features are available.")
        return

    tabs = st.tabs(["Feature patterns", "Nutrients & correlation", "Clustering"])
    with tabs[0]:
        chart_col, distribution_col = st.columns(2)
        with chart_col:
            numeric_cols = numeric.columns.tolist()
            x_default = numeric_cols.index("temperature") if "temperature" in numeric_cols else 0
            y_default = numeric_cols.index("humidity") if "humidity" in numeric_cols else min(1, len(numeric_cols) - 1)
            x = st.selectbox("Horizontal feature", numeric_cols, index=x_default, key="crop_scatter_x")
            y = st.selectbox("Vertical feature", numeric_cols, index=y_default, key="crop_scatter_y")
            fig = px.scatter(
                data,
                x=x,
                y=y,
                color="label" if "label" in data.columns else None,
                title=f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}",
                opacity=0.68,
            )
            st.plotly_chart(chart_layout(fig), width="stretch")
        with distribution_col:
            feature = st.selectbox(
                "Feature distribution",
                numeric_cols,
                index=numeric_cols.index("ph") if "ph" in numeric_cols else 0,
            )
            fig = px.histogram(
                data,
                x=feature,
                color="label" if "label" in data.columns else None,
                marginal="box",
                title=f"Observed {feature} distribution",
                opacity=0.72,
            )
            st.plotly_chart(chart_layout(fig), width="stretch")

    with tabs[1]:
        nutrient_columns = [column for column in ("N", "P", "K") if column in data]
        if nutrient_columns and "label" in data:
            means = data.groupby("label", as_index=False)[nutrient_columns].mean()
            fig = px.bar(
                means,
                x="label",
                y=nutrient_columns,
                barmode="group",
                title="Mean recorded NPK values by crop",
                color_discrete_sequence=["#16835d", "#8cbf9e", "#d4a84e"],
            )
            st.plotly_chart(chart_layout(fig, 420), width="stretch")
        soil_columns = [column for column in ("ph", "soil_moisture", "rainfall") if column in numeric]
        if soil_columns:
            chosen_soil = st.multiselect(
                "Soil and water features",
                soil_columns,
                default=soil_columns,
            )
            if chosen_soil:
                plot_data = data.melt(
                    id_vars=["label"] if "label" in data else None,
                    value_vars=chosen_soil,
                    var_name="Feature",
                    value_name="Observed value",
                )
                fig = px.box(
                    plot_data,
                    x="Feature",
                    y="Observed value",
                    color="label" if "label" in plot_data else None,
                    title="Observed soil, moisture, and rainfall values",
                    points=False,
                )
                st.plotly_chart(chart_layout(fig, 420), width="stretch")
        correlation = numeric.corr()
        heatmap = go.Figure(
            go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                zmin=-1,
                zmax=1,
                colorscale="RdYlGn",
                colorbar_title="r",
            )
        )
        heatmap.update_layout(title="Numeric feature correlation", height=650)
        st.plotly_chart(chart_layout(heatmap, 650), width="stretch")

    with tabs[2]:
        st.caption("K-Means groups numeric, standardized crop/environment features; groups are not crop diagnoses.")
        sample_count = st.slider(
            "Records for clustering",
            min_value=2,
            max_value=min(1200, len(data)),
            value=min(300, len(data)),
            key="crop_cluster_sample",
        )
        clusters = st.slider(
            "Number of K-Means clusters",
            min_value=2,
            max_value=min(10, sample_count),
            value=min(4, sample_count),
            key="crop_cluster_count",
        )
        st.caption(ALGORITHM_NOTES["K-Means"])
        if st.button("Run crop clustering", type="primary"):
            try:
                matrix, metadata, _ = crop_matrix(sample_count, "Numeric fields only")
                projection, labels = run_analysis(matrix, "K-Means", {"clusters": clusters})
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Crop clustering could not run: {exc}")
                return
            results = metadata.copy()
            results["Component 1"] = projection[:, 0]
            results["Component 2"] = projection[:, 1]
            results["Result"] = labels
            results["Algorithm"] = "K-Means"
            results["Dataset"] = "Crop & soil records"
            run_info = save_analysis("Crop & soil records", "K-Means", len(results), results)
            st.success(f"Clustering completed on {len(results):,} actual crop records.")
            st.caption(
                f"Run: {run_info['Completed']} · numeric features only · "
                f"{len(results['Result'].unique())} clusters"
            )
            fig = px.scatter(
                results,
                x="Component 1",
                y="Component 2",
                color="Result",
                title="Crop feature clusters shown in PCA space",
            )
            st.plotly_chart(chart_layout(fig), width="stretch")
            st.dataframe(results, width="stretch", height=300)
            st.download_button(
                "Download crop clustering results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="agrivision_crop_clusters.csv",
                mime="text/csv",
            )
