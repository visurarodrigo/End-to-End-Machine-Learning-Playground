"""Upload CSV and preview data."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import sample_data

st.set_page_config(page_title="Upload Data", layout="wide", initial_sidebar_state="expanded")

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

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📁 Data Upload")
st.caption("Upload your own CSV or load a sample dataset to get started.")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Upload CSV", "Sample Datasets"])

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataset = df
        st.session_state.filename = uploaded_file.name

        st.success(f"**{uploaded_file.name}** uploaded successfully.")

        # ── Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.divider()

        # ── Preview
        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True, key="preview1")

        # ── Column info
        st.subheader("Column Summary")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values,
            "Non-Null Count": df.count().values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(col_info, use_container_width=True, key="colinfo1", hide_index=True)

        # ── Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.warning("Missing values detected — they will be auto-imputed during training.")
            missing_df = missing[missing > 0].reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            st.dataframe(missing_df, use_container_width=True, key="missing1", hide_index=True)
        else:
            st.info("No missing values found.")

        st.divider()

        # ── Next steps
        st.subheader("What's Next?")
        n1, n2, n3 = st.columns(3)
        with n1:
            st.markdown("**📉 Regression**")
            st.caption("Predict a continuous value — works best when your target column is numeric (e.g. price, score).")
        with n2:
            st.markdown("**🎯 Classification**")
            st.caption("Predict a category — works best when your target column is binary or multi-class (e.g. yes/no).")
        with n3:
            st.markdown("**🔍 Unsupervised**")
            st.caption("Find patterns — no target column needed. Great for clustering or reducing dimensions.")

    else:
        st.markdown(" ")
        st.info("No file uploaded yet. Choose a CSV above or try a sample dataset from the next tab.")


# ── Tab 2: Sample Datasets ────────────────────────────────────────────────────
with tab2:
    st.subheader("Sample Datasets")
    st.caption("Click a dataset to load it instantly — no upload needed.")
    st.markdown(" ")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏠 House Prices**")
        st.caption("Predict house sale prices from features like size, location, and number of rooms. Use with the **Regression** page.")
        if st.button("Load House Prices", use_container_width=True, key="btn_regression"):
            df, filename = sample_data.get_sample_regression()
            if df is not None:
                st.session_state.dataset = df
                st.session_state.filename = filename
                st.success("Loaded — head to the Regression page.")
                st.balloons()
            else:
                st.error("Could not load dataset.")

    with col2:
        st.markdown("**💳 Loan Approval**")
        st.caption("Predict whether a loan application is approved or denied based on applicant data. Use with the **Classification** page.")
        if st.button("Load Loan Approval", use_container_width=True, key="btn_classification"):
            df, filename = sample_data.get_sample_classification()
            if df is not None:
                st.session_state.dataset = df
                st.session_state.filename = filename
                st.success("Loaded — head to the Classification page.")
                st.balloons()
            else:
                st.error("Could not load dataset.")

    with col3:
        st.markdown("**🔵 Clustering**")
        st.caption("Discover natural groupings in unlabelled data using KMeans or reduce dimensions with PCA. Use with the **Unsupervised** page.")
        if st.button("Load Clustering Data", use_container_width=True, key="btn_unsupervised"):
            df, filename = sample_data.get_sample_unsupervised()
            if df is not None:
                st.session_state.dataset = df
                st.session_state.filename = filename
                st.success("Loaded — head to the Unsupervised page.")
                st.balloons()
            else:
                st.error("Could not load dataset.")

    st.divider()

    # ── Dataset details
    st.subheader("Dataset Details")
    datasets = sample_data.dataset_info()
    for key, info in datasets.items():
        with st.expander(info["name"], expanded=False):
            st.write(info["description"])
            st.caption(f"Task type: {info['type']}")
            if info["target"]:
                st.caption(f"Target column: {info['target']}")


# ── Current dataset preview (always visible at bottom) ───────────────────────
if "dataset" in st.session_state:
    st.divider()
    st.subheader("Current Dataset")
    st.caption(f"Active file: `{st.session_state.filename}`")

    df = st.session_state.dataset
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown(" ")
    st.dataframe(df, use_container_width=True, key="current_dataset")