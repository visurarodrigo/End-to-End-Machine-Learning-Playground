"""Regression models - Linear, Polynomial, Ridge, Lasso."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import api_client

st.set_page_config(page_title="Regression", layout="wide")

st.title("📉 Regression Models")
st.write("Train multiple regression models and compare performance metrics.")

if "dataset" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first on the Upload page.")
    st.stop()

df = st.session_state.dataset
filename = st.session_state.filename

# Convert dataframe to bytes for API
csv_bytes = BytesIO()
df.to_csv(csv_bytes, index=False)
csv_bytes.seek(0)

st.info(f"📄 Dataset: {filename} ({df.shape[0]} rows × {df.shape[1]} columns)")

# Run regression on upload (this trains all models and gets comparisons)
st.subheader("🔄 Training Models...")

with st.spinner("Training regression models (Linear, Scaled, Polynomial, Ridge, Lasso)..."):
    result = api_client.upload_csv(csv_bytes.getvalue(), filename)

if "error" in result:
    st.error(f"❌ Error: {result['error']}")
    st.stop()

# Extract MSE values
mse_data = {
    "Model": ["Original", "Scaled", "Polynomial", "Ridge", "Lasso"],
    "MSE": [
        result.get("original_mse", 0),
        result.get("scaled_mse", 0),
        result.get("polynomial_mse", 0),
        result.get("ridge_mse", 0),
        result.get("lasso_mse", 0)
    ]
}
mse_df = pd.DataFrame(mse_data).sort_values("MSE")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 Predictions", "📋 Insights"])

with tab1:
    st.subheader("Model Performance Comparison")
    
    # Bar chart
    fig = px.bar(mse_df, x="Model", y="MSE", color="MSE", 
                 color_continuous_scale="Reds", 
                 title="Mean Squared Error (MSE) - Lower is Better")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="regression_mse")
    
    # Table
    st.subheader("Detailed Results")
    comparison = pd.DataFrame({
        "Model": mse_df["Model"],
        "MSE": mse_df["MSE"].round(2),
        "RMSE": (mse_df["MSE"] ** 0.5).round(2)
    })
    
    # Highlight best model
    best_idx = comparison["MSE"].idxmin()
    st.dataframe(
        comparison,
        use_container_width=True,
        key="regression_comparison"
    )

with tab2:
    st.subheader("Actual vs Predicted")
    
    # Sample predictions
    actual = result.get("prediction_analysis", {}).get("actual_values", [])
    predicted = result.get("prediction_analysis", {}).get("predicted_values", [])
    
    if actual and predicted:
        pred_df = pd.DataFrame({
            "Actual": actual,
            "Predicted": predicted,
            "Error": [abs(a - p) for a, p in zip(actual, predicted)]
        })
        pred_df.index = [f"Sample {i+1}" for i in range(len(actual))]
        
        st.dataframe(pred_df, use_container_width=True, key="pred_sample")
        
        # Scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df["Actual"],
            y=pred_df["Predicted"],
            mode="markers",
            marker=dict(size=10, color=pred_df["Error"], colorscale="Viridis", showscale=True),
            text=[f"Error: {e:.2f}" for e in pred_df["Error"]],
            hovertemplate="Actual: %{x}<br>Predicted: %{y}<br>%{text}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[min(pred_df["Actual"]), max(pred_df["Actual"])],
            y=[min(pred_df["Actual"]), max(pred_df["Actual"])],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash", color="red")
        ))
        fig.update_layout(title="Actual vs Predicted Values", 
                         xaxis_title="Actual", yaxis_title="Predicted", height=400)
        st.plotly_chart(fig, use_container_width=True, key="scatter_pred")

with tab3:
    st.subheader("📋 Key Insights")
    
    best_model = mse_df.iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏆 Best Model")
        st.success(f"**{best_model['Model']}**")
        st.metric("MSE", f"{best_model['MSE']:.2f}")
    
    with col2:
        st.markdown(f"### ⚠️ Performance Gap")
        worst_mse = mse_df["MSE"].max()
        improvement = ((worst_mse - best_model['MSE']) / worst_mse * 100)
        st.info(f"**{improvement:.1f}%** improvement over worst model")
    
    st.divider()
    
    st.markdown("### Detailed Messages")
    
    if result.get("scaling_performance_message"):
        st.info(f"**Scaling:** {result['scaling_performance_message']}")
    
    if result.get("polynomial_performance_message"):
        st.info(f"**Polynomial:** {result['polynomial_performance_message']}")
    
    if result.get("regularization_performance_message"):
        st.info(f"**Regularization:** {result['regularization_performance_message']}")
