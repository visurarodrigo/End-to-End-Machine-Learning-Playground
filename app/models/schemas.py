"""Pydantic schemas for API request/response validation and type hints."""

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    """Standard response for classification endpoints."""

    message: str = Field(..., description="Success or status message")
    model: str = Field(..., description="Name of the model trained")
    train_accuracy: float = Field(..., description="Accuracy on training set")
    test_accuracy: float = Field(..., description="Accuracy on test set")
    accuracy_gap: float = Field(..., description="Difference between train and test accuracy (overfitting indicator)")
    precision: float = Field(..., description="Precision score on test set")
    recall: float = Field(..., description="Recall score on test set")
    f1_score: float = Field(..., description="F1 score on test set")
    actual_values: list[int] = Field(..., description="First 10 actual test values")
    predicted_values: list[int] = Field(..., description="First 10 predicted values")
    confusion_matrix: dict = Field(..., description="TP, TN, FP, FN for binary classification")


class ClusteringResponse(BaseModel):
    """Standard response for clustering endpoints."""

    message: str = Field(..., description="Success message")
    model: str = Field(..., description="Model name (e.g., KMeans)")
    k: int = Field(..., description="Number of clusters")
    samples_used: int = Field(..., description="Number of samples processed")
    numeric_columns_used: list[str] = Field(..., description="Feature columns used")
    cluster_labels: list[int] = Field(..., description="Cluster assignment for each sample")
    cluster_centers: list[list[float]] = Field(..., description="Coordinates of cluster centers")
    first_10_cluster_assignments: list[dict] = Field(..., description="First 10 cluster assignments")


class PCAResponse(BaseModel):
    """Standard response for PCA endpoint."""

    message: str = Field(..., description="Success message")
    model: str = Field(..., description="Model name (PCA)")
    n_components: int = Field(..., description="Number of components")
    samples_used: int = Field(..., description="Number of samples processed")
    numeric_columns_used: list[str] = Field(..., description="Feature columns used")
    first_10_transformed_rows: list[list[float]] = Field(..., description="First 10 PCA-transformed rows")
    explained_variance_ratio: list[float] = Field(..., description="Explained variance per component")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status")


class WelcomeResponse(BaseModel):
    """Welcome message response."""

    message: str = Field(..., description="Welcome text")
