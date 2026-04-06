"""Unsupervised learning routes for KMeans clustering and PCA."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.utils.csv_utils import read_csv_upload


router = APIRouter(tags=["unsupervised"])


@router.post("/train-clustering-kmeans", summary="Train a KMeans clustering model")
async def train_clustering_kmeans(
    file: UploadFile = File(...),
    k: int = Form(...),
) -> dict[str, object]:
    """Train a KMeans clustering model from an uploaded CSV file."""
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be a positive integer.")

    df = await read_csv_upload(file)

    X = df.select_dtypes(include="number")
    if X.empty:
        raise HTTPException(status_code=400, detail="No numeric columns found in CSV after dropping non-numeric columns.")

    if X.isnull().any().any():
        raise HTTPException(status_code=400, detail="Numeric feature columns contain missing values. Please clean missing values first.")

    if k > len(X):
        raise HTTPException(status_code=400, detail=f"k cannot be greater than number of samples ({len(X)}).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)

    cluster_labels = model.labels_.tolist()
    first_10_assignments = [
        {"row_index": int(index), "cluster": int(label)}
        for index, label in zip(X.index[:10], cluster_labels[:10])
    ]

    return {
        "message": "KMeans clustering training completed successfully.",
        "model": "KMeans",
        "k": k,
        "samples_used": int(len(X)),
        "numeric_columns_used": [str(column) for column in X.columns.tolist()],
        "cluster_labels": [int(label) for label in cluster_labels],
        "cluster_centers": model.cluster_centers_.tolist(),
        "first_10_cluster_assignments": first_10_assignments,
    }


@router.post("/train-pca", summary="Apply PCA dimensionality reduction")
async def train_pca(
    file: UploadFile = File(...),
    n_components: int = Form(...),
) -> dict[str, object]:
    """Apply PCA on numeric columns from an uploaded CSV file."""
    if n_components <= 0:
        raise HTTPException(status_code=400, detail="n_components must be a positive integer.")

    df = await read_csv_upload(file)

    X = df.select_dtypes(include="number")
    if X.empty:
        raise HTTPException(status_code=400, detail="No numeric columns found in CSV for PCA.")

    if X.isnull().any().any():
        raise HTTPException(status_code=400, detail="Numeric columns contain missing values. Please clean missing values first.")

    max_components = min(X.shape[0], X.shape[1])
    if n_components > max_components:
        raise HTTPException(
            status_code=400,
            detail=(
                "n_components is too large. "
                f"It must be <= min(n_samples, n_features) which is {max_components}."
            ),
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    transformed_first_10 = X_pca[:10].tolist()

    return {
        "message": "PCA completed successfully.",
        "model": "PCA",
        "n_components": n_components,
        "samples_used": int(X.shape[0]),
        "numeric_columns_used": [str(column) for column in X.columns.tolist()],
        "first_10_transformed_rows": transformed_first_10,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
