# End-to-End Machine Learning Playground

A production-style, API-first machine learning playground built to help you learn, test, and compare ML workflows from data ingestion to model evaluation.

## Overview

End-to-End Machine Learning Playground is a hands-on FastAPI project for experimenting with both supervised and unsupervised machine learning in a practical way.

You upload CSV data, train models through HTTP endpoints, and receive structured outputs including predictions, evaluation metrics, clustering assignments, and dimensionality-reduced representations.

This project is designed for:

- Students learning ML concepts through implementation
- Developers practicing MLOps-friendly API design
- Engineers who want a reusable ML experimentation backend

## Features

### 1. Regression

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- MSE-based model comparison

### 2. Classification

- Logistic Regression (pipeline with scaling)
- Decision Tree Classifier (supports max_depth tuning)
- Random Forest Classifier

### 3. Neural Network (Optional)

- TensorFlow/Keras binary classification endpoint

### 4. Clustering

- K-Means clustering on scaled numeric features
- Returns cluster labels and centers

### 5. Dimensionality Reduction

- PCA on scaled numeric features
- Returns transformed rows and explained variance ratio

### 6. Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Train vs test comparison
- Accuracy gap for overfitting detection

### 7. Data Processing

- CSV upload via API
- Automatic numeric-column selection where needed
- Missing-value checks and handling flow
- Feature scaling with StandardScaler
- Dataset preview and schema validation

### 8. Clean Architecture

- Route-driven endpoint organization
- Service layer for model training logic
- Reusable evaluation helpers
- Extensible structure for future ML modules

## Tech Stack

- Backend: FastAPI, Uvicorn
- Machine Learning: Scikit-learn
- Data Processing: Pandas, NumPy
- Neural Networks (optional): TensorFlow / Keras
- Language: Python 3.10+

## Project Structure

```text
End to End Machine Learning Playground/
├─ README.md
├─ requirements.txt
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ routes/
│  │  ├─ __init__.py
│  │  └─ classification.py
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ model_service.py
│  │  └─ evaluation_service.py
│  ├─ utils/
│  │  └─ __init__.py
│  └─ models/
│     └─ __init__.py
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
│  └─ README.md
├─ notebooks/
│  └─ README.md
└─ tests/
	└─ README.md
```

## Installation Guide

1. Clone the repository

```bash
git clone <your-repo-url>
cd "End to End Machine Learning Playground"
```

2. Create a virtual environment

```bash
python -m venv .venv
```

3. Activate the environment

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Default local URL:

- API: http://127.0.0.1:8000
- Swagger Docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints

### Core

- `GET /` - Welcome message
- `GET /health` - Health check

### Data + Regression

- `POST /upload` - Upload CSV, inspect dataset, and run regression/model comparison workflow

### Classification

- `POST /train-classification-logistic`
- `POST /train-classification-decision-tree`
- `POST /train-classification-random-forest`
- `POST /train-classification-neural-network` (optional TensorFlow)

### Unsupervised Learning

- `POST /train-clustering-kmeans` - K-Means clustering
- `POST /train-pca` - PCA dimensionality reduction

## Example API Usage

### 1. Train Logistic Regression

```bash
curl -X POST "http://127.0.0.1:8000/train-classification-logistic" \
	-F "file=@data/raw/sample_classification.csv" \
	-F "target_column=target"
```

Sample response:

```json
{
	"message": "Logistic regression training completed successfully.",
	"model": "LogisticRegression",
	"train_accuracy": 0.91,
	"test_accuracy": 0.88,
	"accuracy_gap": 0.03,
	"precision": 0.87,
	"recall": 0.89,
	"f1_score": 0.88,
	"actual_values": [1, 0, 1, 1, 0],
	"predicted_values": [1, 0, 1, 0, 0],
	"confusion_matrix": {
		"true_negatives": 42,
		"false_positives": 6,
		"false_negatives": 5,
		"true_positives": 47
	}
}
```

### 2. Train K-Means Clustering

```bash
curl -X POST "http://127.0.0.1:8000/train-clustering-kmeans" \
	-F "file=@data/raw/sample_unsupervised.csv" \
	-F "k=3"
```

Sample response:

```json
{
	"message": "KMeans clustering training completed successfully.",
	"model": "KMeans",
	"k": 3,
	"samples_used": 150,
	"numeric_columns_used": ["feature_1", "feature_2", "feature_3"],
	"cluster_labels": [0, 1, 0, 2, 1],
	"cluster_centers": [[0.52, -0.14, 1.01], [-0.88, 0.34, -0.22], [1.20, 0.77, -0.56]],
	"first_10_cluster_assignments": [
		{"row_index": 0, "cluster": 0},
		{"row_index": 1, "cluster": 1}
	]
}
```

### 3. Apply PCA

```bash
curl -X POST "http://127.0.0.1:8000/train-pca" \
	-F "file=@data/raw/sample_unsupervised.csv" \
	-F "n_components=2"
```

Sample response:

```json
{
	"message": "PCA completed successfully.",
	"model": "PCA",
	"n_components": 2,
	"samples_used": 150,
	"numeric_columns_used": ["feature_1", "feature_2", "feature_3"],
	"first_10_transformed_rows": [[1.42, -0.37], [0.98, 0.14]],
	"explained_variance_ratio": [0.61, 0.27]
}
```

## Learning Outcomes

By building and using this project, you will learn how to:

- Design ML systems as APIs using FastAPI
- Build reproducible preprocessing and training workflows
- Compare multiple models using consistent metrics
- Detect overfitting via train/test performance gap
- Work with both supervised and unsupervised techniques
- Organize ML code using route/service architecture
- Prepare a portfolio-quality end-to-end ML backend

## Future Improvements

- Add model persistence and loading (joblib/pickle)
- Add cross-validation and hyperparameter search
- Add multi-class and imbalanced-dataset support
- Add authentication and rate limiting for APIs
- Add experiment tracking (MLflow or Weights & Biases)
- Add Docker and CI/CD pipelines
- Expand test coverage with integration and contract tests
- Add frontend dashboard for dataset upload and result visualization

## Author

Visura Rodrigo
