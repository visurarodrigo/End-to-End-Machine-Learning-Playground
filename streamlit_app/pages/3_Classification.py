"""Classification models - Logistic, Decision Tree, Random Forest, Neural Network."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import api_client

st.set_page_config(page_title="Classification", layout="wide")

st.title("🎯 Classification Models")
st.write("Train and compare classification models on your dataset.")

if "dataset" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first on the Upload page.")
    st.stop()

df = st.session_state.dataset
filename = st.session_state.filename

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
target_column = st.sidebar.selectbox(
    "Target Column (for prediction)",
    numeric_cols,
    help="Select the column you want to predict"
)

models = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Neural Network": "neural_network"
}

selected_models = st.sidebar.multiselect(
    "Select Models to Train",
    list(models.keys()),
    default=["Logistic Regression", "Decision Tree", "Random Forest"]
)

# Optional: Decision tree depth
max_depth = None
if "Decision Tree" in selected_models:
    max_depth = st.sidebar.slider("Decision Tree Max Depth", 1, 20, 5)

# Convert dataframe to bytes
csv_bytes = BytesIO()
df.to_csv(csv_bytes, index=False)
csv_bytes.seek(0)

st.info(f"📄 Dataset: {filename} | Target: {target_column} | Selected Models: {len(selected_models)}")

# Train models
if st.button("🚀 Train Selected Models", use_container_width=True):
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(selected_models)
    
    for idx, model_name in enumerate(selected_models):
        status_text.text(f"Training {model_name}...")
        
        # Convert to bytes fresh for each request
        csv_bytes_fresh = BytesIO()
        df.to_csv(csv_bytes_fresh, index=False)
        csv_bytes_fresh.seek(0)
        
        if model_name == "Logistic Regression":
            result = api_client.train_logistic_regression(
                csv_bytes_fresh.getvalue(), filename, target_column
            )
        elif model_name == "Decision Tree":
            result = api_client.train_decision_tree(
                csv_bytes_fresh.getvalue(), filename, target_column, max_depth
            )
        elif model_name == "Random Forest":
            result = api_client.train_random_forest(
                csv_bytes_fresh.getvalue(), filename, target_column
            )
        elif model_name == "Neural Network":
            result = api_client.train_neural_network(
                csv_bytes_fresh.getvalue(), filename, target_column
            )
        
        results[model_name] = result
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    # Store results
    st.session_state.classification_results = results
    st.success("✅ Training complete!")

# Display results if available
if "classification_results" in st.session_state:
    results = st.session_state.classification_results
    
    tab1, tab2, tab3 = st.tabs(["📊 Comparison", "📈 Details", "❌ Confusion Matrices"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        # Prepare comparison data
        comparison_data = []
        for model_name, result in results.items():
            if "error" not in result:
                comparison_data.append({
                    "Model": model_name,
                    "Train Accuracy": result.get("train_accuracy", 0),
                    "Test Accuracy": result.get("test_accuracy", 0),
                    "Precision": result.get("precision", 0),
                    "Recall": result.get("recall", 0),
                    "F1 Score": result.get("f1_score", 0)
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Bar chart - Accuracy comparison
            fig = px.bar(comp_df, x="Model", y=["Train Accuracy", "Test Accuracy"],
                        barmode="group", title="Model Accuracy Comparison",
                        color_discrete_map={"Train Accuracy": "#1f77b4", "Test Accuracy": "#ff7f0e"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="class_accuracy")
            
            st.dataframe(
                comparison_data,
                use_container_width=True,
                key="class_comparison"
            )
    
    with tab2:
        st.subheader("Detailed Metrics Per Model")
        
        for model_name, result in results.items():
            if "error" not in result:
                with st.expander(f"📋 {model_name}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Train Accuracy", f"{result.get('train_accuracy', 0):.4f}")
                    with col2:
                        st.metric("Test Accuracy", f"{result.get('test_accuracy', 0):.4f}")
                    with col3:
                        gap = result.get("accuracy_gap", 0)
                        st.metric("Accuracy Gap", f"{gap:.4f}", 
                                 delta_color="inverse" if gap > 0.1 else "normal")
                    with col4:
                        st.metric("F1 Score", f"{result.get('f1_score', 0):.4f}")
                    
                    st.divider()
                    
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric("Precision", f"{result.get('precision', 0):.4f}")
                    with col6:
                        st.metric("Recall", f"{result.get('recall', 0):.4f}")
                    
                    # Sample predictions
                    st.subheader("Sample Predictions")
                    if result.get("actual_values") and result.get("predicted_values"):
                        pred_sample = pd.DataFrame({
                            "Actual": result["actual_values"],
                            "Predicted": result["predicted_values"]
                        })
                        st.dataframe(pred_sample, use_container_width=True, key=f"pred_{model_name}")
            else:
                st.error(f"❌ {model_name}: {result['error']}")
    
    with tab3:
        st.subheader("Confusion Matrices")
        
        for model_name, result in results.items():
            if "error" not in result and "confusion_matrix" in result:
                cm = result["confusion_matrix"]
                
                with st.expander(f"🔲 {model_name}", expanded=False):
                    # Create confusion matrix heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=[[cm.get("TP", 0), cm.get("FP", 0)],
                           [cm.get("FN", 0), cm.get("TN", 0)]],
                        x=["Predicted Positive", "Predicted Negative"],
                        y=["Actual Positive", "Actual Negative"],
                        text=[[f"TP: {cm.get('TP', 0)}", f"FP: {cm.get('FP', 0)}"],
                              [f"FN: {cm.get('FN', 0)}", f"TN: {cm.get('TN', 0)}"]],
                        texttemplate="%{text}",
                        colorscale="Blues"
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key=f"cm_{model_name}")
