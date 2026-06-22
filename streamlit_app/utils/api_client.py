"""ML service layer — calls backend logic directly (no FastAPI required)."""

import sys
import os
from io import BytesIO

import pandas as pd

# Make sure the project root is on the path so we can import from app/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.regression_service import run_regression_workflow
from app.services.model_service import (
    train_logistic_classifier,
    train_decision_tree_classifier,
    train_random_forest_classifier,
)
from app.services.evaluation_service import build_classification_metrics_response
from sklearn.model_selection import train_test_split


def health_check():
    """Always returns True — no backend server needed."""
    return True


def _load_csv(file_bytes) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame."""
    return pd.read_csv(BytesIO(file_bytes))


def _prepare_classification_data(df: pd.DataFrame, target_column: str):
    """Split DataFrame into train/test sets for classification."""
    if target_column not in df.columns:
        return None, None, None, None, f"Target column '{target_column}' not found."

    X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, None


def upload_csv(file_bytes, filename):
    """Run regression workflow directly on uploaded CSV."""
    try:
        df = _load_csv(file_bytes)
        result = run_regression_workflow(df)
        result["filename"] = filename
        return result
    except Exception as e:
        return {"error": str(e)}


def train_logistic_regression(file_bytes, filename, target_column):
    """Train logistic regression directly."""
    try:
        df = _load_csv(file_bytes)
        X_train, X_test, y_train, y_test, err = _prepare_classification_data(df, target_column)
        if err:
            return {"error": err}
        y_train_pred, y_test_pred = train_logistic_classifier(X_train, y_train, X_test)
        return build_classification_metrics_response(
            message="Logistic Regression training complete.",
            model_name="Logistic Regression",
            y_train=y_train, y_train_pred=y_train_pred,
            y_test=y_test, y_test_pred=y_test_pred,
        )
    except Exception as e:
        return {"error": str(e)}


def train_decision_tree(file_bytes, filename, target_column, max_depth=None):
    """Train decision tree directly."""
    try:
        df = _load_csv(file_bytes)
        X_train, X_test, y_train, y_test, err = _prepare_classification_data(df, target_column)
        if err:
            return {"error": err}
        y_train_pred, y_test_pred = train_decision_tree_classifier(
            X_train, y_train, X_test, max_depth=max_depth
        )
        return build_classification_metrics_response(
            message="Decision Tree training complete.",
            model_name="Decision Tree",
            y_train=y_train, y_train_pred=y_train_pred,
            y_test=y_test, y_test_pred=y_test_pred,
        )
    except Exception as e:
        return {"error": str(e)}


def train_random_forest(file_bytes, filename, target_column):
    """Train random forest directly."""
    try:
        df = _load_csv(file_bytes)
        X_train, X_test, y_train, y_test, err = _prepare_classification_data(df, target_column)
        if err:
            return {"error": err}
        y_train_pred, y_test_pred = train_random_forest_classifier(X_train, y_train, X_test)
        return build_classification_metrics_response(
            message="Random Forest training complete.",
            model_name="Random Forest",
            y_train=y_train, y_train_pred=y_train_pred,
            y_test=y_test, y_test_pred=y_test_pred,
        )
    except Exception as e:
        return {"error": str(e)}


def train_neural_network(file_bytes, filename, target_column):
    """
    Neural network placeholder.
    TensorFlow is too heavy for Streamlit Cloud free tier.
    Returns a clear message instead of crashing.
    """
    return {
        "error": "Neural network training is not available in the cloud demo. "
                 "Run the project locally with the full FastAPI backend to use this feature."
    }


def train_kmeans(file_bytes, filename, k):
    """Train KMeans clustering directly."""
    try:
        from sklearn.cluster import KMeans
        import numpy as np

        df = _load_csv(file_bytes)
        X = df.select_dtypes(include=["number"]).dropna()

        if X.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns for clustering."}

        model = KMeans(n_clusters=int(k), random_state=42, n_init=10)
        labels = model.fit_predict(X)

        return {
            "message": f"KMeans clustering complete with k={k}.",
            "n_clusters": int(k),
            "inertia": float(model.inertia_),
            "cluster_labels": labels.tolist(),
            "cluster_centers": model.cluster_centers_.tolist(),
            "columns_used": X.columns.tolist(),
            "sample_data": X.head(50).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


def train_pca(file_bytes, filename, n_components):
    """Run PCA directly."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = _load_csv(file_bytes)
        X = df.select_dtypes(include=["number"]).dropna()

        n_components = int(n_components)
        if n_components > X.shape[1]:
            return {"error": f"n_components ({n_components}) cannot exceed number of numeric columns ({X.shape[1]})."}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        return {
            "message": f"PCA complete with {n_components} components.",
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "transformed_data": X_pca.tolist(),
            "original_columns": X.columns.tolist(),
        }
    except Exception as e:
        return {"error": str(e)}