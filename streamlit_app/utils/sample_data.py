"""Load sample datasets for quick testing."""

import pandas as pd
from pathlib import Path


def get_sample_regression():
    """Load sample regression dataset (house prices)."""
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_regression.csv"
    try:
        df = pd.read_csv(data_path)
        return df, "sample_regression.csv"
    except FileNotFoundError:
        return None, None


def get_sample_classification():
    """Load sample classification dataset (loan approval)."""
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_classification.csv"
    try:
        df = pd.read_csv(data_path)
        return df, "sample_classification.csv"
    except FileNotFoundError:
        return None, None


def get_sample_unsupervised():
    """Load sample unsupervised dataset (clusters)."""
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_unsupervised.csv"
    try:
        df = pd.read_csv(data_path)
        return df, "sample_unsupervised.csv"
    except FileNotFoundError:
        return None, None


def dataset_info():
    """Return metadata about datasets."""
    return {
        "regression": {
            "name": "🏠 House Price Prediction",
            "description": "Linear & polynomial regression models for real estate valuation",
            "target": "price",
            "type": "Regression",
            "loader": get_sample_regression
        },
        "classification": {
            "name": "💳 Loan Approval Prediction",
            "description": "Classification models for credit risk assessment",
            "target": "target",
            "type": "Classification",
            "loader": get_sample_classification
        },
        "unsupervised": {
            "name": "🎯 Customer Clustering",
            "description": "Explore patterns with KMeans clustering and PCA",
            "target": None,
            "type": "Unsupervised",
            "loader": get_sample_unsupervised
        }
    }
