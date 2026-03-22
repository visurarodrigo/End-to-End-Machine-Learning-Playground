"""Test script to validate all features."""
import requests
import pandas as pd
from io import BytesIO

def test_api():
    """Test all API endpoints."""
    
    # Test 1: Health check
    print("=" * 60)
    print("TEST 1: API Health Check")
    print("=" * 60)
    resp = requests.get('http://127.0.0.1:8000/health')
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
    
    # Test 2: Regression
    print("=" * 60)
    print("TEST 2: Regression")
    print("=" * 60)
    df_reg = pd.read_csv('data/raw/sample_regression.csv')
    print(f"Loaded {len(df_reg)} rows, {len(df_reg.columns)} columns")
    
    csv_bytes = BytesIO()
    df_reg.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    resp = requests.post(
        'http://127.0.0.1:8000/upload',
        files={'file': ('sample_regression.csv', csv_bytes, 'text/csv')},
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Original MSE: {data.get('original_mse', 0):.4f}")
        print(f"Best Model: {data.get('model_comparison', {}).get('best_model_by_mse', 'N/A')}")
        print("✅ Regression works!\n")
    else:
        print(f"❌ Error: {resp.text[:200]}\n")
    
    # Test 3: Classification - Logistic
    print("=" * 60)
    print("TEST 3: Classification - Logistic Regression")
    print("=" * 60)
    df_class = pd.read_csv('data/raw/sample_classification.csv')
    print(f"Loaded {len(df_class)} rows")
    print(f"Target unique values: {sorted(df_class['target'].unique())}")
    
    csv_bytes = BytesIO()
    df_class.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    resp = requests.post(
        'http://127.0.0.1:8000/train-classification-logistic',
        files={'file': ('sample_classification.csv', csv_bytes, 'text/csv')},
        data={'target_column': 'target'},
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Model: {data.get('model', 'N/A')}")
        print(f"Test Accuracy: {data.get('test_accuracy', 0):.4f}")
        print(f"F1 Score: {data.get('f1_score', 0):.4f}")
        print("✅ Logistic Regression works!\n")
    else:
        print(f"❌ Error: {resp.text[:200]}\n")
    
    # Test 4: Classification - Decision Tree
    print("=" * 60)
    print("TEST 4: Classification - Decision Tree")
    print("=" * 60)
    csv_bytes = BytesIO()
    df_class.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    resp = requests.post(
        'http://127.0.0.1:8000/train-classification-decision-tree',
        files={'file': ('sample_classification.csv', csv_bytes, 'text/csv')},
        data={'target_column': 'target', 'max_depth': '5'},
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Model: {data.get('model', 'N/A')}")
        print(f"Test Accuracy: {data.get('test_accuracy', 0):.4f}")
        print("✅ Decision Tree works!\n")
    else:
        print(f"❌ Error: {resp.text[:200]}\n")
    
    # Test 5: Clustering
    print("=" * 60)
    print("TEST 5: KMeans Clustering")
    print("=" * 60)
    df_unsup = pd.read_csv('data/raw/sample_unsupervised.csv')
    print(f"Loaded {len(df_unsup)} rows, {len(df_unsup.columns)} columns")
    
    csv_bytes = BytesIO()
    df_unsup.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    resp = requests.post(
        'http://127.0.0.1:8000/train-clustering-kmeans',
        files={'file': ('sample_unsupervised.csv', csv_bytes, 'text/csv')},
        data={'k': '3'},
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Clusters: {data.get('k', 0)}")
        print(f"Samples: {data.get('samples_used', 0)}")
        print("✅ KMeans works!\n")
    else:
        print(f"❌ Error: {resp.text[:200]}\n")
    
    # Test 6: PCA
    print("=" * 60)
    print("TEST 6: PCA Dimensionality Reduction")
    print("=" * 60)
    csv_bytes = BytesIO()
    df_unsup.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    resp = requests.post(
        'http://127.0.0.1:8000/train-pca',
        files={'file': ('sample_unsupervised.csv', csv_bytes, 'text/csv')},
        data={'n_components': '2'},
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Components: {data.get('n_components', 0)}")
        variance = data.get('explained_variance_ratio', [])
        if variance:
            print(f"Cumulative Variance: {sum(variance)*100:.1f}%")
        print("✅ PCA works!\n")
    else:
        print(f"❌ Error: {resp.text[:200]}\n")
    
    print("=" * 60)
    print("ALL TESTS COMPLETED ✅")
    print("=" * 60)

if __name__ == "__main__":
    test_api()
