"""Manual smoke test for API endpoints.

Run this script directly when the FastAPI server is already running.
"""

import io

import pandas as pd
import requests


BASE_URL = "http://127.0.0.1:8000"


def run_smoke_test() -> None:
    """Exercise the main API endpoints against a running backend."""
    print("=" * 60)
    print("API SMOKE TEST")
    print("=" * 60)

    print("\n1. Health endpoint")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n2. Root endpoint")
    response = requests.get(f"{BASE_URL}/", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n3. Logistic regression endpoint")
    data = {
        "age": [25, 35, 45, 30, 28, 40, 32, 38],
        "income": [50000, 75000, 100000, 60000, 55000, 90000, 65000, 70000],
        "credit_score": [750, 800, 720, 680, 700, 780, 700, 760],
        "loan_amount": [10000, 50000, 100000, 30000, 25000, 75000, 40000, 60000],
        "target": [1, 1, 0, 0, 1, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = requests.post(
        f"{BASE_URL}/train-classification-logistic",
        files={"file": ("test.csv", csv_buffer, "text/csv")},
        data={"target_column": "target"},
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n4. KMeans endpoint")
    import numpy as np

    np.random.seed(42)
    cluster1 = np.random.randn(10, 3) + np.array([0, 0, 0])
    cluster2 = np.random.randn(10, 3) + np.array([5, 5, 5])
    clustering_df = pd.DataFrame(np.vstack([cluster1, cluster2]), columns=["f1", "f2", "f3"])
    csv_buffer = io.BytesIO()
    clustering_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = requests.post(
        f"{BASE_URL}/train-clustering-kmeans",
        files={"file": ("test.csv", csv_buffer, "text/csv")},
        data={"k": "2"},
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n5. PCA endpoint")
    csv_buffer = io.BytesIO()
    clustering_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = requests.post(
        f"{BASE_URL}/train-pca",
        files={"file": ("test.csv", csv_buffer, "text/csv")},
        data={"n_components": "2"},
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    run_smoke_test()
