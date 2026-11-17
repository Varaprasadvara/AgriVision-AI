import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import chart_layout
from dashboard.data import crop_continuous_columns, crop_data, data_locations
from dashboard.theme import page_intro


FEATURE_DESCRIPTIONS = {
    "N": "Recorded soil nitrogen value.",
    "P": "Recorded soil phosphorus value.",
    "K": "Recorded soil potassium value.",
    "temperature": "Recorded ambient temperature.",
    "humidity": "Recorded relative humidity.",
    "ph": "Recorded soil pH.",
    "rainfall": "Recorded rainfall.",
    "label": "Crop label supplied with the dataset.",
    "soil_type": "Integer-coded soil category; a category-name mapping is not included.",
    "water_source_type": "Integer-coded water-source category; a category-name mapping is not included.",
    "growth_stage": "Integer-coded growth-stage category; a category-name mapping is not included.",
}


def render():
    page_intro(
        "DATA MANAGEMENT",
        "Crop Dataset",
        "Inspect the original crop and environmental CSV, its fields, missing values, and distributions.",
    )
    try:
        data = crop_data()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Crop CSV could not be loaded: {exc}")
        return

    st.caption(f"Source: {data_locations()['Crop CSV']}")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{data.shape[0]:,}")
    b.metric("Columns", f"{data.shape[1]:,}")
    c.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")

    st.subheader("Records")
    filters = st.columns([1.3, 1, 1])
    query = filters[0].text_input("Search values", key="crop_dataset_search").strip().casefold()
    filtered = data
    if "label" in data.columns:
        labels = sorted(data["label"].dropna().astype(str).unique())
        selected_labels = filters[1].multiselect("Crop labels", labels, default=labels)
        filtered = filtered[filtered["label"].astype(str).isin(selected_labels)]
    if query:
        found = filtered.astype(str).apply(
            lambda column: column.str.casefold().str.contains(query, regex=False, na=False)
        ).any(axis=1)
        filtered = filtered.loc[found]
    filters[2].caption(f"{len(filtered):,} matching rows")
    st.dataframe(filtered.head(500), width="stretch", height=370)
    st.download_button(
        "Download original crop CSV",
        data.to_csv(index=False).encode("utf-8"),
        file_name="Crop_recommendationV2.csv",
        mime="text/csv",
    )

    with st.expander("Field names, types, and descriptions", expanded=True):
        fields = pd.DataFrame(
            {
                "Field": data.columns,
                "Data type": [str(dtype) for dtype in data.dtypes],
                "Missing": data.isna().sum().values,
                "Description": [FEATURE_DESCRIPTIONS.get(column, "Additional dataset field; meaning is not specified in the source files.") for column in data.columns],
            }
        )
        st.dataframe(fields, width="stretch", hide_index=True)
    with st.expander("Missing values and descriptive statistics"):
        missing = data.isna().sum().rename("Missing values").to_frame()
        missing["Missing %"] = (missing["Missing values"] / max(1, len(data)) * 100).round(2)
        st.dataframe(missing, width="stretch")
        st.dataframe(data.describe(include="all").transpose(), width="stretch")

    numeric = data[crop_continuous_columns(data)]
    if not numeric.empty:
        st.subheader("Feature distributions")
        feature = st.selectbox("Numeric feature", numeric.columns.tolist())
        fig = px.histogram(
            data,
            x=feature,
            color="label" if "label" in data.columns else None,
            marginal="box",
            title=f"Distribution of {feature}",
            opacity=0.76,
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
        st.subheader("Numeric feature correlation")
        st.caption("Integer-coded soil, water-source, and growth-stage categories are excluded from Pearson correlation.")
        correlation = numeric.corr()
        heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Pearson correlation across numeric fields",
        )
        st.plotly_chart(chart_layout(heatmap, 620), width="stretch")
    st.caption("Tables and charts summarize observed CSV values; they are not model outputs.")
