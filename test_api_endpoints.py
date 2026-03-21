"""Script to test API endpoints."""

import requests
import pandas as pd
import io
import json

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("📊 TESTING API ENDPOINTS")
print("=" * 60)

# Test 1: Health Check
print("\n1️⃣  Testing Health Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Root Endpoint
print("\n2️⃣  Testing Root Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Classification Endpoint
print("\n3️⃣  Testing Classification Endpoint (Logistic Regression)...")
try:
    # Create sample data
    data = {
        'age': [25, 35, 45, 30, 28, 40, 32, 38],
        'income': [50000, 75000, 100000, 60000, 55000, 90000, 65000, 70000],
        'credit_score': [750, 800, 720, 680, 700, 780, 700, 760],
        'loan_amount': [10000, 50000, 100000, 30000, 25000, 75000, 40000, 60000],
        'target': [1, 1, 0, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    data_form = {'target_column': 'target'}
    
    response = requests.post(
        f"{BASE_URL}/train-classification-logistic",
        files=files,
        data=data_form
    )
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Model: {result.get('model')}")
    print(f"   Train Accuracy: {result.get('train_accuracy'):.4f}")
    print(f"   Test Accuracy: {result.get('test_accuracy'):.4f}")
    print(f"   F1 Score: {result.get('f1_score'):.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Decision Tree Endpoint
print("\n4️⃣  Testing Decision Tree Classifier...")
try:
    # Reuse same data
    data = {
        'age': [25, 35, 45, 30, 28, 40, 32, 38],
        'income': [50000, 75000, 100000, 60000, 55000, 90000, 65000, 70000],
        'credit_score': [750, 800, 720, 680, 700, 780, 700, 760],
        'loan_amount': [10000, 50000, 100000, 30000, 25000, 75000, 40000, 60000],
        'target': [1, 1, 0, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    data_form = {'target_column': 'target', 'max_depth': '5'}
    
    response = requests.post(
        f"{BASE_URL}/train-classification-decision-tree",
        files=files,
        data=data_form
    )
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Model: {result.get('model')}")
    print(f"   Train Accuracy: {result.get('train_accuracy'):.4f}")
    print(f"   Test Accuracy: {result.get('test_accuracy'):.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Random Forest Endpoint
print("\n5️⃣  Testing Random Forest Classifier...")
try:
    data = {
        'age': [25, 35, 45, 30, 28, 40, 32, 38],
        'income': [50000, 75000, 100000, 60000, 55000, 90000, 65000, 70000],
        'credit_score': [750, 800, 720, 680, 700, 780, 700, 760],
        'loan_amount': [10000, 50000, 100000, 30000, 25000, 75000, 40000, 60000],
        'target': [1, 1, 0, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    data_form = {'target_column': 'target'}
    
    response = requests.post(
        f"{BASE_URL}/train-classification-random-forest",
        files=files,
        data=data_form
    )
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Model: {result.get('model')}")
    print(f"   Train Accuracy: {result.get('train_accuracy'):.4f}")
    print(f"   Test Accuracy: {result.get('test_accuracy'):.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 6: KMeans Clustering
print("\n6️⃣  Testing KMeans Clustering Endpoint...")
try:
    import numpy as np
    np.random.seed(42)
    cluster1 = np.random.randn(10, 3) + np.array([0, 0, 0])
    cluster2 = np.random.randn(10, 3) + np.array([5, 5, 5])
    X = pd.DataFrame(np.vstack([cluster1, cluster2]), columns=['f1', 'f2', 'f3'])
    
    csv_buffer = io.BytesIO()
    X.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    data_form = {'k': '2'}
    
    response = requests.post(
        f"{BASE_URL}/train-clustering-kmeans",
        files=files,
        data=data_form
    )
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Model: {result.get('model')}")
    print(f"   K: {result.get('k')}")
    print(f"   Samples: {result.get('samples_used')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 7: PCA Dimensionality Reduction
print("\n7️⃣  Testing PCA Endpoint...")
try:
    import numpy as np
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    
    csv_buffer = io.BytesIO()
    X.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test.csv', csv_buffer, 'text/csv')}
    data_form = {'n_components': '2'}
    
    response = requests.post(
        f"{BASE_URL}/train-pca",
        files=files,
        data=data_form
    )
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Model: {result.get('model')}")
    print(f"   Components: {result.get('n_components')}")
    print(f"   Samples: {result.get('samples_used')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ API ENDPOINT TESTING COMPLETE!")
print("=" * 60)
