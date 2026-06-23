"""Main Streamlit App - ML Playground."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="ML Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .feature-header { font-size: 1rem; font-weight: 600; margin-bottom: 0.25rem; }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🧪 ML Playground")
st.markdown(
    "An interactive machine learning environment — upload a dataset, "
    "train models, and compare results across four learning tasks."
)
st.divider()

# ── What you can do ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**📁 Upload**")
    st.caption("Bring your own CSV or load a built-in sample dataset to get started instantly.")

with col2:
    st.markdown("**📉 Regression**")
    st.caption("Compare Linear, Polynomial, Ridge, and Lasso models on continuous targets.")

with col3:
    st.markdown("**🎯 Classification**")
    st.caption("Train Logistic Regression, Decision Tree, and Random Forest classifiers.")

with col4:
    st.markdown("**🔍 Unsupervised**")
    st.caption("Explore KMeans clustering and PCA dimensionality reduction.")

st.divider()

# ── How to use ──────────────────────────────────────────────────────────────
st.subheader("Getting Started")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("**Step 1**")
    st.caption("Go to **Upload** in the sidebar. Upload your CSV or click a sample dataset button.")

with step2:
    st.markdown("**Step 2**")
    st.caption("Navigate to the learning task that fits your data — Regression, Classification, or Unsupervised.")

with step3:
    st.markdown("**Step 3**")
    st.caption("Select a model, adjust parameters if needed, and click Train.")

with step4:
    st.markdown("**Step 4**")
    st.caption("Review metrics, charts, and model comparisons to understand your results.")

st.divider()

# ── Models at a glance ──────────────────────────────────────────────────────
st.subheader("Models Available")

r_col, c_col, u_col = st.columns(3)

with r_col:
    st.markdown("**Regression**")
    st.caption("Linear · Scaled Linear · Polynomial · Ridge · Lasso")

with c_col:
    st.markdown("**Classification**")
    st.caption("Logistic Regression · Decision Tree · Random Forest")

with u_col:
    st.markdown("**Unsupervised**")
    st.caption("KMeans Clustering · Principal Component Analysis (PCA)")

st.divider()

# ── Footer ──────────────────────────────────────────────────────────────────
st.caption("Built with Streamlit · scikit-learn · pandas · plotly")