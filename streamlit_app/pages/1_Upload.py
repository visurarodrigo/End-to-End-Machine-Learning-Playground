"""Upload CSV and preview data."""

import streamlit as st
import pandas as pd
from io import BytesIO
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import sample_data, api_client

st.set_page_config(page_title="Upload Data", layout="wide", initial_sidebar_state="expanded")

st.title("📁 Upload Your Dataset")

# Check API health
if not api_client.health_check():
    st.error("⚠️ **API not running!** Make sure FastAPI is running at http://127.0.0.1:8000")
    st.info("Start the API with: `uvicorn app.main:app --reload`")
    st.stop()

st.write("Upload a CSV file or try one of our sample datasets to explore ML workflows.")

# Tabs for upload vs samples
tab1, tab2 = st.tabs(["Upload CSV", "Sample Datasets"])

with tab1:
    st.subheader("Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read file
        df = pd.read_csv(uploaded_file)
        
        # Store in session state for other pages
        st.session_state.dataset = df
        st.session_state.filename = uploaded_file.name
        
        # Preview
        st.success("✅ File uploaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True, key="preview1")
        
        st.subheader("Column Info")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes,
            "Non-Null": df.count(),
            "Unique": df.nunique()
        })
        st.dataframe(col_info, use_container_width=True, key="colinfo1")
        
        # Show missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"⚠️ Missing values detected:\n{missing[missing > 0]}")
        else:
            st.info("✅ No missing values found!")
        
        st.divider()
        st.subheader("Next Steps")
        st.markdown("""
        - **Regression:** Try the Regression tab to predict continuous values
        - **Classification:** Try the Classification tab for binary prediction
        - **Unsupervised:** Try the Unsupervised tab for clustering & dimensionality reduction
        """)


with tab2:
    st.subheader("🚀 Quick Start with Sample Datasets")
    
    datasets = sample_data.dataset_info()
    
    col1, col2, col3 = st.columns(3)
    
    # Regression sample
    with col1:
        if st.button("🏠 House Prices", use_container_width=True, key="btn_regression"):
            df, filename = sample_data.get_sample_regression()
            if df is not None:
                st.session_state.dataset = df
                st.session_state.filename = filename
                st.success("✅ Loaded! Check the Regression page")
                st.balloons()
            else:
                st.error("❌ Could not load sample dataset")
    
    # Classification sample
    with col2:
        if st.button("💳 Loan Approval", use_container_width=True, key="btn_classification"):
            df, filename = sample_data.get_sample_classification()
            if df is not None:
                st.session_state.dataset = df
                st.session_state.filename = filename
                st.success("✅ Loaded! Check the Classification page")
                st.balloons()
            else:
                st.error("❌ Could not load sample dataset")
    
    # Unsupervised sample
    with col3:
        if st.button("🎯 Clustering", use_container_width=True, key="btn_unsupervised"):
            df, filename = sample_data.get_sample_unsupervised()
            if df is not None:
                st.session_state.dataset = df
                st.session_state.filename = filename
                st.success("✅ Loaded! Check the Unsupervised page")
                st.balloons()
            else:
                st.error("❌ Could not load sample dataset")
    
    st.divider()
    
    # Dataset descriptions
    for key, info in datasets.items():
        with st.expander(f"{info['name']}", expanded=False):
            st.write(info['description'])
            st.caption(f"Type: {info['type']}")
            if info['target']:
                st.caption(f"Target: {info['target']}")

# Show current dataset
if "dataset" in st.session_state:
    st.divider()
    st.subheader("📊 Current Dataset")
    st.caption(f"Filename: {st.session_state.filename}")
    st.dataframe(st.session_state.dataset, use_container_width=True, key="current_dataset")
