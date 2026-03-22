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

st.set_page_config(page_title="Unsupervised", layout="wide")

st.title("🔍 Unsupervised Learning")
st.write("Explore patterns with KMeans Clustering and PCA dimensionality reduction.")

if "dataset" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first on the Upload page.")
    st.stop()

df = st.session_state.dataset
filename = st.session_state.filename

# Get numeric columns
numeric_df = df.select_dtypes(include=['number'])

if numeric_df.empty:
    st.error("❌ No numeric columns found. Please upload a dataset with numeric features.")
    st.stop()

st.info(f"📄 Dataset: {filename} | Numeric Features: {len(numeric_df.columns)}")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

tab1, tab2 = st.tabs(["🎯 KMeans Clustering", "📉 PCA Reduction"])

with tab1:
    st.subheader("KMeans Clustering")
    
    k = st.slider("Number of Clusters (k)", min_value=2, max_value=min(10, len(df)), value=3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run KMeans", use_container_width=True):
            with st.spinner(f"Training KMeans with k={k}..."):
                csv_bytes = BytesIO()
                df.to_csv(csv_bytes, index=False)
                csv_bytes.seek(0)
                
                result = api_client.train_kmeans(csv_bytes.getvalue(), filename, k)
            
            if "error" not in result:
                st.session_state.kmeans_result = result
                st.success("✅ KMeans training complete!")
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # Display KMeans results
    if "kmeans_result" in st.session_state:
        result = st.session_state.kmeans_result
        
        cluster_labels = result.get("cluster_labels", [])
        
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clusters", k)
        with col2:
            st.metric("Samples", len(cluster_labels))
        with col3:
            st.metric("Features Used", len(result.get("numeric_columns_used", [])))
        
        # Cluster distribution
        cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_dist.index,
            y=cluster_dist.values,
            title="Cluster Distribution",
            labels={"x": "Cluster", "y": "Number of Samples"}
        )
        st.plotly_chart(fig, use_container_width=True, key="cluster_dist")
        
        # First 20 cluster assignments
        st.subheader("Sample Cluster Assignments (First 20)")
        assignments = result.get("first_10_cluster_assignments", [])
        if assignments:
            assign_df = pd.DataFrame(assignments)
            st.dataframe(assign_df, use_container_width=True, key="cluster_assignments")

with tab2:
    st.subheader("PCA Dimensionality Reduction")
    
    max_components = min(numeric_df.shape[0], numeric_df.shape[1])
    n_components = st.slider(
        "Number of Components",
        min_value=2,
        max_value=max_components,
        value=min(3, max_components)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run PCA", use_container_width=True):
            with st.spinner(f"Training PCA with {n_components} components..."):
                csv_bytes = BytesIO()
                df.to_csv(csv_bytes, index=False)
                csv_bytes.seek(0)
                
                result = api_client.train_pca(csv_bytes.getvalue(), filename, n_components)
            
            if "error" not in result:
                st.session_state.pca_result = result
                st.success("✅ PCA training complete!")
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # Display PCA results
    if "pca_result" in st.session_state:
        result = st.session_state.pca_result
        
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Components", n_components)
        with col2:
            st.metric("Samples", result.get("samples_used", 0))
        with col3:
            st.metric("Original Features", len(result.get("numeric_columns_used", [])))
        
        # Explained variance
        variance = result.get("explained_variance_ratio", [])
        cumsum_variance = np.cumsum(variance)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"PC {i+1}" for i in range(len(variance))],
            y=variance,
            name="Individual",
            marker_color="lightblue"
        ))
        fig.add_trace(go.Scatter(
            x=[f"PC {i+1}" for i in range(len(variance))],
            y=cumsum_variance,
            name="Cumulative",
            yaxis="y2",
            mode="lines+markers",
            marker_color="darkblue",
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Explained Variance by Component",
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained",
            yaxis2=dict(title="Cumulative Variance", overlaying="y", side="right"),
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, key="pca_variance")
        
        # Variance percentage table
        st.subheader("Variance Explained by Component")
        variance_df = pd.DataFrame({
            "Component": [f"PC {i+1}" for i in range(len(variance))],
            "Variance (%)": [f"{v*100:.2f}%" for v in variance],
            "Cumulative (%)": [f"{c*100:.2f}%" for c in cumsum_variance]
        })
        st.dataframe(variance_df, use_container_width=True, key="variance_table")
        
        st.info(f"💡 First {n_components} components explain **{cumsum_variance[-1]*100:.1f}%** of total variance")
