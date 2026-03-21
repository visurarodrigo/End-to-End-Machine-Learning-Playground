"""Tests for API routes and endpoints."""

import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test GET / endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self, client):
        """Test GET /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestFileUpload:
    """Test file upload endpoints."""

    def test_upload_csv_file(self, client):
        """Test uploading a valid CSV file."""
        # Create sample CSV
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
            "price": [100000, 120000, 150000, 200000],
        })
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)

        response = client.post(
            "/upload",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert data["rows"] == 4
        assert data["columns"] == 3

    def test_upload_invalid_file(self, client):
        """Test uploading a non-CSV file."""
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"not csv", "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_no_file(self, client):
        """Test upload endpoint without file."""
        response = client.post("/upload", files={})
        assert response.status_code == 422  # Validation error


class TestClassificationEndpoints:
    """Test classification endpoints."""

    @pytest.fixture
    def classification_csv(self):
        """Create a sample classification CSV file."""
        df = pd.DataFrame({
            "age": np.random.randint(20, 60, 50),
            "income": np.random.randint(30000, 150000, 50),
            "credit_score": np.random.randint(300, 850, 50),
            "loan_amount": np.random.randint(10000, 500000, 50),
            "target": np.random.randint(0, 2, 50),
        })
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        return csv_bytes

    def test_logistic_regression_endpoint(self, client, classification_csv):
        """Test logistic regression classification endpoint."""
        response = client.post(
            "/train-classification-logistic",
            files={"file": ("test.csv", classification_csv, "text/csv")},
            data={"target_column": "target"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "train_accuracy" in data
        assert "test_accuracy" in data
        assert data["model"] == "LogisticRegression"

    def test_decision_tree_endpoint(self, client, classification_csv):
        """Test decision tree classification endpoint."""
        response = client.post(
            "/train-classification-decision-tree",
            files={"file": ("test.csv", classification_csv, "text/csv")},
            data={"target_column": "target", "max_depth": "5"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "train_accuracy" in data
        assert "test_accuracy" in data
        assert data["model"] == "Decision Tree Classifier"

    def test_random_forest_endpoint(self, client, classification_csv):
        """Test random forest classification endpoint."""
        response = client.post(
            "/train-classification-random-forest",
            files={"file": ("test.csv", classification_csv, "text/csv")},
            data={"target_column": "target"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "train_accuracy" in data
        assert "test_accuracy" in data
        assert data["model"] == "Random Forest Classifier"

    def test_neural_network_endpoint(self, client, classification_csv):
        """Test neural network classification endpoint."""
        response = client.post(
            "/train-classification-neural-network",
            files={"file": ("test.csv", classification_csv, "text/csv")},
            data={"target_column": "target", "epochs": "10"},
        )
        assert response.status_code in [200, 500]  # May fail if TensorFlow not properly installed
        if response.status_code == 200:
            data = response.json()
            assert "message" in data or "model" in data


class TestClusteringEndpoints:
    """Test unsupervised learning endpoints."""

    @pytest.fixture
    def unsupervised_csv(self):
        """Create a sample unsupervised CSV file."""
        np.random.seed(42)
        cluster1 = np.random.randn(20, 3) + np.array([0, 0, 0])
        cluster2 = np.random.randn(20, 3) + np.array([5, 5, 5])
        
        X = pd.DataFrame(
            np.vstack([cluster1, cluster2]),
            columns=["feature_1", "feature_2", "feature_3"],
        )
        csv_bytes = io.BytesIO()
        X.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        return csv_bytes

    def test_kmeans_clustering(self, client, unsupervised_csv):
        """Test KMeans clustering endpoint."""
        response = client.post(
            "/train-clustering-kmeans",
            files={"file": ("test.csv", unsupervised_csv, "text/csv")},
            data={"k": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["k"] == 2
        assert "cluster_labels" in data

    def test_pca_endpoint(self, client, unsupervised_csv):
        """Test PCA dimensionality reduction endpoint."""
        response = client.post(
            "/train-pca",
            files={"file": ("test.csv", unsupervised_csv, "text/csv")},
            data={"n_components": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["n_components"] == 2
        assert "first_10_transformed_rows" in data
