import numpy as np
import plotly.express as px
import streamlit as st


def empty_state(message):
    st.info(message)


def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=32, t=72, b=112),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            size=13,
            color="#102B24",
        ),
        title=dict(
            font=dict(size=17, color="#102B24"),
            x=0.02,
            xanchor="left",
        ),
        legend=dict(
            font=dict(size=12, color="#24372D"),
            bgcolor="rgba(255,255,255,.92)",
            bordercolor="#DDE7E0",
            borderwidth=1,
        ),
        legend_title_text="",
    )
    fig.update_xaxes(
        title_font=dict(size=13, color="#24372D"),
        tickfont=dict(size=11, color="#34483A"),
        automargin=True,
        gridcolor="#E8EEEA",
        linecolor="#B7C9BE",
        zerolinecolor="#DDE7E0",
    )
    fig.update_yaxes(
        title_font=dict(size=13, color="#24372D"),
        tickfont=dict(size=11, color="#34483A"),
        automargin=True,
        gridcolor="#E8EEEA",
        linecolor="#B7C9BE",
        zerolinecolor="#DDE7E0",
    )
    fig.update_annotations(font=dict(color="#24372D", size=12))
    for trace in fig.data:
        values = getattr(trace, "x", None)
        if values is not None and len(values):
            labels = np.asarray(values)
            if labels.dtype.kind in "OUS" and len(np.unique(labels)) > 8:
                fig.update_xaxes(tickangle=-35)
                break
    return fig


def pseudo_rgb(image, red_band=40, green_band=20, blue_band=10):
    """Create a display composite from band indices, without wavelength claims."""
    channels = [image[:, :, index].astype(np.float32) for index in (red_band, green_band, blue_band)]
    rgb = np.stack(channels, axis=-1)
    output = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[:, :, channel]
        low, high = np.nanpercentile(plane, [2, 98])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            output[:, :, channel] = np.clip((plane - low) / (high - low), 0, 1)
    return output


def scatter_plot(frame, x, y, color, title, hover_data=None):
    fig = px.scatter(
        frame,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        title=title,
        opacity=0.78,
    )
    return chart_layout(fig)
