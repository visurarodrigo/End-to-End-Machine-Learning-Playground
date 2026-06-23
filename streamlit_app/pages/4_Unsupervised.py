"""Unsupervised Learning - KMeans Clustering and PCA."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import api_client

st.set_page_config(page_title="Unsupervised Learning", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Unsupervised Learning")
st.caption("Discover patterns in your data with KMeans Clustering and PCA dimensionality reduction.")
st.divider()

# ── Dataset check ─────────────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.info("No dataset loaded. Go to the **Upload** page first.")
    st.stop()

df = st.session_state.dataset
filename = st.session_state.filename
numeric_df = df.select_dtypes(include=["number"])

if numeric_df.empty:
    st.error("No numeric columns found. Please upload a dataset with numeric features.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Dataset", filename)
c2.metric("Rows", f"{df.shape[0]:,}")
c3.metric("Numeric Features", len(numeric_df.columns))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["KMeans Clustering", "PCA Reduction"])

# ── Tab 1: KMeans ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("KMeans Clustering")
    st.caption("Groups your data into k clusters based on similarity across numeric features.")
    st.markdown(" ")

    k = st.slider("Number of Clusters (k)", min_value=2, max_value=min(10, len(df)), value=3)

    if st.button("Run KMeans", use_container_width=True, type="primary", key="run_kmeans"):
        with st.spinner(f"Clustering into {k} groups..."):
            csv_bytes = BytesIO()
            df.to_csv(csv_bytes, index=False)
            csv_bytes.seek(0)
            result = api_client.train_kmeans(csv_bytes.getvalue(), filename, k)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.session_state.kmeans_result = result
            st.success("Clustering complete.")

    if "kmeans_result" in st.session_state:
        result = st.session_state.kmeans_result
        cluster_labels = result.get("cluster_labels", [])
        columns_used = result.get("columns_used", [])

        st.divider()

        # ── Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Clusters", k)
        m2.metric("Samples Clustered", f"{len(cluster_labels):,}")
        m3.metric("Inertia", f"{result.get('inertia', 0):,.2f}")

        st.caption(
            "Inertia measures how tightly packed each cluster is — "
            "lower is better, but always decreases as k increases."
        )

        st.divider()

        # ── Cluster distribution
        st.subheader("Cluster Distribution")
        cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()

        fig = px.bar(
            x=[f"Cluster {i}" for i in cluster_dist.index],
            y=cluster_dist.values,
            labels={"x": "Cluster", "y": "Samples"},
            text=cluster_dist.values,
            color=cluster_dist.values,
            color_continuous_scale="Blues",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=340,
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#e9ecef"),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="cluster_dist")

        # ── 2D scatter if enough features
        sample_data = result.get("sample_data", [])
        if sample_data and len(columns_used) >= 2:
            st.divider()
            st.subheader("Cluster Scatter Plot")
            st.caption(f"Plotting first two features: **{columns_used[0]}** vs **{columns_used[1]}**")

            scatter_df = pd.DataFrame(sample_data)
            scatter_df["Cluster"] = [str(l) for l in cluster_labels[:len(scatter_df)]]

            fig2 = px.scatter(
                scatter_df,
                x=columns_used[0],
                y=columns_used[1],
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(
                height=380,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#e9ecef"),
                yaxis=dict(showgrid=True, gridcolor="#e9ecef"),
                margin=dict(t=10, b=10),
                legend=dict(title="Cluster"),
            )
            st.plotly_chart(fig2, use_container_width=True, key="cluster_scatter")

# ── Tab 2: PCA ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("PCA Dimensionality Reduction")
    st.caption("Reduces many features into a smaller set of components that capture the most variance.")
    st.markdown(" ")

    max_components = min(numeric_df.shape[0], numeric_df.shape[1])
    n_components = st.slider(
        "Number of Components",
        min_value=2,
        max_value=max_components,
        value=min(3, max_components),
    )

    if st.button("Run PCA", use_container_width=True, type="primary", key="run_pca"):
        with st.spinner(f"Running PCA with {n_components} components..."):
            csv_bytes = BytesIO()
            df.to_csv(csv_bytes, index=False)
            csv_bytes.seek(0)
            result = api_client.train_pca(csv_bytes.getvalue(), filename, n_components)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.session_state.pca_result = result
            st.success("PCA complete.")

    if "pca_result" in st.session_state:
        result = st.session_state.pca_result
        variance = result.get("explained_variance_ratio", [])
        cumsum_variance = np.cumsum(variance).tolist()
        total_variance = result.get("total_variance_explained", cumsum_variance[-1] if cumsum_variance else 0)

        st.divider()

        # ── Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Components", n_components)
        m2.metric("Original Features", len(result.get("original_columns", [])))
        m3.metric("Variance Explained", f"{total_variance * 100:.1f}%")

        st.divider()

        # ── Variance chart
        st.subheader("Explained Variance by Component")
        st.caption("Individual bars show variance per component. The line shows cumulative variance.")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(variance))],
            y=[round(v * 100, 2) for v in variance],
            name="Individual",
            marker_color="#adb5bd",
            text=[f"{v*100:.1f}%" for v in variance],
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(variance))],
            y=[round(c * 100, 2) for c in cumsum_variance],
            name="Cumulative",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ))
        fig.update_layout(
            height=380,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, title="Component"),
            yaxis=dict(
                showgrid=True, gridcolor="#e9ecef",
                title="Variance (%)", range=[0, max([v * 100 for v in variance]) * 1.3]
            ),
            yaxis2=dict(
                title="Cumulative (%)",
                overlaying="y", side="right",
                range=[0, 110],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="pca_variance")

        st.divider()

        # ── Variance table
        st.subheader("Component Breakdown")
        variance_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(variance))],
            "Variance Explained (%)": [f"{v * 100:.2f}%" for v in variance],
            "Cumulative (%)": [f"{c * 100:.2f}%" for c in cumsum_variance],
        })
        st.dataframe(variance_df, use_container_width=True, hide_index=True, key="variance_table")

        # ── 2D scatter of first two PCs
        transformed = result.get("transformed_data", [])
        if transformed and n_components >= 2:
            st.divider()
            st.subheader("PC1 vs PC2 Scatter")
            st.caption("Each point is one row from your dataset, projected onto the first two principal components.")

            pca_plot_df = pd.DataFrame(
                [[row[0], row[1]] for row in transformed],
                columns=["PC1", "PC2"]
            )
            fig2 = px.scatter(
                pca_plot_df, x="PC1", y="PC2",
                opacity=0.6,
                color_discrete_sequence=["#1f77b4"],
            )
            fig2.update_layout(
                height=360,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#e9ecef", title="PC1"),
                yaxis=dict(showgrid=True, gridcolor="#e9ecef", title="PC2"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True, key="pca_scatter")