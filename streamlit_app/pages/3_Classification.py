"""Classification models - Logistic, Decision Tree, Random Forest."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import api_client

st.set_page_config(page_title="Classification", layout="wide")

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
st.title("🎯 Classification")
st.caption("Train and compare classification models on your dataset.")
st.divider()

# ── Dataset check ─────────────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.info("No dataset loaded. Go to the **Upload** page first.")
    st.stop()

df = st.session_state.dataset
filename = st.session_state.filename

# ── Sidebar config ────────────────────────────────────────────────────────────
st.sidebar.markdown("### Configuration")

all_cols = df.columns.tolist()
target_column = st.sidebar.selectbox(
    "Target Column",
    all_cols,
    help="The column you want to predict."
)

selected_models = st.sidebar.multiselect(
    "Models to Train",
    ["Logistic Regression", "Decision Tree", "Random Forest"],
    default=["Logistic Regression", "Decision Tree", "Random Forest"],
)

max_depth = None
if "Decision Tree" in selected_models:
    max_depth = st.sidebar.slider("Decision Tree — Max Depth", 1, 20, 5)

# ── Dataset summary ───────────────────────────────────────────────────────────
target_series = df[target_column]
n_classes = target_series.nunique(dropna=False)
majority_baseline = float(target_series.value_counts(dropna=False).max() / len(target_series))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", filename)
c2.metric("Rows", f"{df.shape[0]:,}")
c3.metric("Target Classes", n_classes)
c4.metric("Baseline Accuracy", f"{majority_baseline:.3f}")

if n_classes > 20:
    st.warning(
        "This target column has many unique values and may not suit classification. "
        "Consider choosing a binary or categorical column."
    )

st.divider()

# ── Config change detection ───────────────────────────────────────────────────
config_signature = (filename, target_column, tuple(sorted(selected_models)), max_depth)
if st.session_state.get("classification_config_signature") != config_signature:
    st.session_state.pop("classification_results", None)

# ── Train button ──────────────────────────────────────────────────────────────
if st.button("Train Models", use_container_width=True, type="primary"):
    if not selected_models:
        st.warning("Select at least one model from the sidebar.")
        st.stop()

    results = {}
    progress = st.progress(0)
    status = st.empty()
    total = len(selected_models)

    for idx, model_name in enumerate(selected_models):
        status.caption(f"Training {model_name}...")

        csv_fresh = BytesIO()
        df.to_csv(csv_fresh, index=False)
        csv_fresh.seek(0)
        b = csv_fresh.getvalue()

        if model_name == "Logistic Regression":
            result = api_client.train_logistic_regression(b, filename, target_column)
        elif model_name == "Decision Tree":
            result = api_client.train_decision_tree(b, filename, target_column, max_depth)
        elif model_name == "Random Forest":
            result = api_client.train_random_forest(b, filename, target_column)

        results[model_name] = result
        progress.progress((idx + 1) / total)

    progress.empty()
    status.empty()

    st.session_state.classification_results = results
    st.session_state.classification_config_signature = config_signature
    st.success("Training complete.")

# ── Results ───────────────────────────────────────────────────────────────────
if "classification_results" not in st.session_state:
    st.stop()

results = st.session_state.classification_results

# ── Build comparison table ────────────────────────────────────────────────────
comparison_data = []
for model_name, result in results.items():
    if "error" not in result:
        comparison_data.append({
            "Model": model_name,
            "Train Accuracy": round(result.get("train_accuracy", 0), 4),
            "Test Accuracy": round(result.get("test_accuracy", 0), 4),
            "Accuracy Gap": round(result.get("accuracy_gap", 0), 4),
            "Precision": round(result.get("precision", 0), 4),
            "Recall": round(result.get("recall", 0), 4),
            "F1 Score": round(result.get("f1_score", 0), 4),
        })

# ── Summary metrics ───────────────────────────────────────────────────────────
if comparison_data:
    comp_df = pd.DataFrame(comparison_data)
    best_row = comp_df.loc[comp_df["Test Accuracy"].idxmax()]

    s1, s2, s3 = st.columns(3)
    s1.metric("Best Model", best_row["Model"])
    s2.metric("Best Test Accuracy", f"{best_row['Test Accuracy']:.4f}")
    s3.metric("Best F1 Score", f"{comp_df['F1 Score'].max():.4f}")
    st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Comparison", "Per-Model Details", "Confusion Matrices"])

# ── Tab 1: Comparison ─────────────────────────────────────────────────────────
with tab1:
    if not comparison_data:
        st.info("No results to display yet. Train models above.")
    else:
        st.subheader("Accuracy — Train vs Test")
        st.caption("A large gap between train and test accuracy suggests overfitting.")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp_df["Model"],
            y=comp_df["Train Accuracy"],
            name="Train Accuracy",
            marker_color="#adb5bd",
        ))
        fig.add_trace(go.Bar(
            x=comp_df["Model"],
            y=comp_df["Test Accuracy"],
            name="Test Accuracy",
            marker_color="#1f77b4",
        ))
        fig.update_layout(
            barmode="group",
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True, gridcolor="#e9ecef",
                range=[0, 1], title="Accuracy"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="class_accuracy")

        st.divider()
        st.subheader("Full Metrics Table")
        st.dataframe(comp_df, use_container_width=True, hide_index=True, key="class_comparison")

# ── Tab 2: Per-model details ──────────────────────────────────────────────────
with tab2:
    for model_name, result in results.items():
        if "error" in result:
            st.error(f"{model_name}: {result['error']}")
            continue

        with st.expander(model_name, expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train Accuracy", f"{result.get('train_accuracy', 0):.4f}")
            m2.metric("Test Accuracy", f"{result.get('test_accuracy', 0):.4f}")
            gap = result.get("accuracy_gap", 0)
            m3.metric("Accuracy Gap", f"{gap:.4f}",
                      delta=f"{'overfitting risk' if gap > 0.1 else 'ok'}",
                      delta_color="inverse" if gap > 0.1 else "off")
            m4.metric("F1 Score", f"{result.get('f1_score', 0):.4f}")

            st.markdown(" ")
            p1, p2 = st.columns(2)
            p1.metric("Precision", f"{result.get('precision', 0):.4f}")
            p2.metric("Recall", f"{result.get('recall', 0):.4f}")

            if result.get("actual_values") and result.get("predicted_values"):
                st.markdown(" ")
                st.caption("Sample predictions (first 10 from test set)")
                pred_df = pd.DataFrame({
                    "Actual": result["actual_values"],
                    "Predicted": result["predicted_values"],
                })
                pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]
                st.dataframe(pred_df, use_container_width=True,
                             hide_index=True, key=f"pred_{model_name}")

# ── Tab 3: Confusion matrices ─────────────────────────────────────────────────
with tab3:
    st.caption("Confusion matrices show where your model makes correct vs incorrect predictions.")

    for model_name, result in results.items():
        if "error" in result or "confusion_matrix" not in result:
            continue

        cm = result["confusion_matrix"]

        with st.expander(model_name, expanded=False):
            if "matrix" in cm:
                st.caption("Multi-class confusion matrix")
                st.dataframe(
                    pd.DataFrame(cm["matrix"]),
                    use_container_width=True,
                    key=f"cm_table_{model_name}"
                )
            else:
                tp = cm.get("true_positives", 0)
                fp = cm.get("false_positives", 0)
                fn = cm.get("false_negatives", 0)
                tn = cm.get("true_negatives", 0)

                fig = go.Figure(data=go.Heatmap(
                    z=[[tp, fp], [fn, tn]],
                    x=["Predicted Positive", "Predicted Negative"],
                    y=["Actual Positive", "Actual Negative"],
                    text=[[f"TP: {tp}", f"FP: {fp}"],
                          [f"FN: {fn}", f"TN: {tn}"]],
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                ))
                fig.update_layout(
                    height=320,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"cm_{model_name}")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("True Positives", tp)
                r2.metric("False Positives", fp)
                r3.metric("False Negatives", fn)
                r4.metric("True Negatives", tn)