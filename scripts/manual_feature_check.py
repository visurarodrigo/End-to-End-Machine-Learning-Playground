"""Manual feature check for the API using the bundled sample datasets."""

from io import BytesIO

import pandas as pd
import requests


def test_api() -> None:
    """Run a small end-to-end validation of the backend endpoints."""
    print("=" * 60)
    print("FEATURE CHECK")
    print("=" * 60)

    resp = requests.get("http://127.0.0.1:8000/health", timeout=10)
    print(f"Health: {resp.status_code} {resp.json()}")

    df_reg = pd.read_csv("data/raw/sample_regression.csv")
    csv_bytes = BytesIO()
    df_reg.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    resp = requests.post(
        "http://127.0.0.1:8000/upload",
        files={"file": ("sample_regression.csv", csv_bytes, "text/csv")},
        timeout=30,
    )
    print(f"Regression: {resp.status_code}")

    df_class = pd.read_csv("data/raw/sample_classification.csv")
    csv_bytes = BytesIO()
    df_class.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    resp = requests.post(
        "http://127.0.0.1:8000/train-classification-logistic",
        files={"file": ("sample_classification.csv", csv_bytes, "text/csv")},
        data={"target_column": "target"},
        timeout=30,
    )
    print(f"Classification: {resp.status_code}")

    df_unsup = pd.read_csv("data/raw/sample_unsupervised.csv")
    csv_bytes = BytesIO()
    df_unsup.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    resp = requests.post(
        "http://127.0.0.1:8000/train-clustering-kmeans",
        files={"file": ("sample_unsupervised.csv", csv_bytes, "text/csv")},
        data={"k": "3"},
        timeout=30,
    )
    print(f"Clustering: {resp.status_code}")


if __name__ == "__main__":
    test_api()
