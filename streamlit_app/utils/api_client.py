"""API client for communicating with FastAPI backend."""

import requests
import pandas as pd
from io import BytesIO

API_BASE_URL = "http://127.0.0.1:8000"


def health_check():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_csv(file_bytes, filename):
    """Upload CSV file to regression endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_logistic_regression(file_bytes, filename, target_column):
    """Train logistic regression classifier."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/train-classification-logistic",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"target_column": target_column},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_decision_tree(file_bytes, filename, target_column, max_depth=None):
    """Train decision tree classifier."""
    try:
        data = {"target_column": target_column}
        if max_depth is not None:
            data["max_depth"] = max_depth
        
        response = requests.post(
            f"{API_BASE_URL}/train-classification-decision-tree",
            files={"file": (filename, file_bytes, "text/csv")},
            data=data,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_random_forest(file_bytes, filename, target_column):
    """Train random forest classifier."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/train-classification-random-forest",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"target_column": target_column},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_neural_network(file_bytes, filename, target_column):
    """Train neural network classifier."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/train-classification-neural-network",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"target_column": target_column},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_kmeans(file_bytes, filename, k):
    """Train KMeans clustering model."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/train-clustering-kmeans",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"k": k},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_pca(file_bytes, filename, n_components):
    """Train PCA dimensionality reduction."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/train-pca",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"n_components": n_components},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}
