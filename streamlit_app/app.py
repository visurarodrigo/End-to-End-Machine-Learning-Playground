"""Main Streamlit App - ML Playground."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import api_client

st.set_page_config(
    page_title="ML Playground",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🚀 ML Playground")
st.sidebar.markdown("---")

# Check API status
api_running = api_client.health_check()
if api_running:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline")

st.sidebar.markdown("---")

# Main content
st.title("🤖 End-to-End Machine Learning Playground")
st.markdown("""
### Welcome! 👋

This is your interactive ML lab where you can:

1. **📁 Upload or Sample** - Bring your CSV or test with built-in datasets
2. **📉 Regression** - Compare Linear, Polynomial, Ridge, and Lasso models
3. **🎯 Classification** - Train Logistic, Decision Tree, Random Forest, and Neural Networks
4. **🔍 Unsupervised** - Explore KMeans Clustering and PCA

### Quick Start

1. Go to **Upload** page → Load a dataset (or try a sample)
2. Pick your learning task → See results instantly
3. Compare models and get recommendations

---
""")

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📁 Upload Data
    - Drag & drop CSV
    - Auto data preview
    - Sample datasets ready
    """)

with col2:
    st.markdown("""
    ### 📊 Train Models
    - Instant training
    - Live metrics
    - Visual comparisons
    """)

with col3:
    st.markdown("""
    ### 📈 Get Insights
    - Performance metrics
    - Confusion matrices
    - Variance analysis
    """)

st.divider()

# Instructions
with st.expander("📚 How to Use (Click to expand)", expanded=False):
    st.markdown("""
    #### Step 1: Upload Your Dataset
    - Navigate to the **Upload** page
    - Either upload your CSV or click a sample dataset button
    - Preview your data and check for issues
    
    #### Step 2: Choose a Task
    - **Regression** → for predicting continuous values (prices, temperature, etc.)
    - **Classification** → for binary predictions (yes/no, approved/denied, etc.)
    - **Unsupervised** → for finding patterns (clusters, dimensionality reduction)
    
    #### Step 3: Configure & Run
    - Select models you want to train
    - Adjust parameters (if needed)
    - Click "Train" and watch results appear
    
    #### Step 4: Analyze Results
    - Compare model performance
    - Review metrics and visualizations
    - Get recommendations for best model
    """)

st.divider()

# API Status
st.subheader("API Status")
if api_running:
    st.success("✅ **API is running** at http://127.0.0.1:8000")
    st.caption("Make sure your FastAPI backend is active: `uvicorn app.main:app --reload`")
else:
    st.error("❌ **API is not responding**")
    st.warning("""
    Start your FastAPI backend:
    ```bash
    cd "End to End Machine Learning Playground"
    uvicorn app.main:app --reload
    ```
    Then refresh this page.
    """)

st.divider()

# Footer
st.markdown("""
---
**Built for learning ML workflows from data ingestion to model evaluation.**

Made with ❤️ using Streamlit + FastAPI
""")
