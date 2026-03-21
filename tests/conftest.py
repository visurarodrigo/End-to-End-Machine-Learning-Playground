"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 4),
        columns=["feature_1", "feature_2", "feature_3", "feature_4"],
    )
    y = pd.Series(np.random.randint(0, 2, 100), name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 3),
        columns=["feature_1", "feature_2", "feature_3"],
    )
    y = pd.Series(
        3 * X["feature_1"] + 2 * X["feature_2"] - X["feature_3"] + np.random.randn(100),
        name="target",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def sample_unsupervised_data():
    """Create sample unsupervised dataset for testing."""
    np.random.seed(42)
    # Create clusters
    cluster1 = np.random.randn(30, 3) + np.array([0, 0, 0])
    cluster2 = np.random.randn(30, 3) + np.array([5, 5, 5])
    cluster3 = np.random.randn(30, 3) + np.array([-5, 5, -5])
    
    X = pd.DataFrame(
        np.vstack([cluster1, cluster2, cluster3]),
        columns=["feature_1", "feature_2", "feature_3"],
    )
    return {"X": X}
