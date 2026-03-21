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
├─ README.md                      # Project documentation, setup, and API usage guide
├─ requirements.txt               # Python dependency list (production dependencies)
├─ requirements-dev.txt           # Development dependencies (pytest, black, flake8, etc.)
├─ config.py                      # Application configuration management (paths, settings)
├─ pytest.ini                     # Pytest configuration and test discovery settings
├─ .env                           # Environment variables template
├─ .gitignore                     # Git ignore rules
├─ PROJECT_FINALIZATION.md        # Project completion summary and documentation
├─ test_api_endpoints.py          # Manual API endpoint testing script
├─ app/                           # Application source code
│  ├─ README.md                   # Overview of app modules and responsibilities
│  ├─ __init__.py                 # Marks app as a Python package
│  ├─ main.py                     # FastAPI app entry point and non-classification routes
│  ├─ FastAPI - Swagger UI.png    # Swagger UI screenshot reference
│  ├─ routes/                     # API route definitions
│  │  ├─ README.md                # Notes about route modules and usage
│  │  ├─ __init__.py              # Marks routes as a Python package
│  │  └─ classification.py        # Classification-related endpoints (4 models)
│  ├─ services/                   # Reusable business/ML logic
│  │  ├─ README.md                # Service-layer design and extension notes
│  │  ├─ __init__.py              # Marks services as a Python package
│  │  ├─ model_service.py         # Model training helpers for sklearn classifiers
│  │  └─ evaluation_service.py    # Metrics: accuracy, precision, recall, F1, confusion matrix
│  ├─ utils/                      # Shared utility helpers (expand as project grows)
│  │  ├─ README.md                # Utility helper guidance and conventions
│  │  └─ __init__.py              # Marks utils as a Python package
│  └─ models/                     # App-level model schemas/types (Pydantic)
│     ├─ __init__.py              # Marks models as a Python package
│     ├─ README.md                # Notes for app schemas and typed payloads
│     └─ schemas.py               # Pydantic response schemas (ClassificationResponse, etc.)
├─ data/                          # Local datasets for experiments
│  ├─ README.md                   # Data folder purpose and organization
│  ├─ raw/                        # Original, unprocessed datasets
│  │  ├─ README.md                # Dataset descriptions and quick-start guide
│  │  ├─ sample_regression.csv    # Sample regression dataset for /upload endpoint
│  │  ├─ sample_classification.csv# Sample binary classification dataset
│  │  └─ sample_unsupervised.csv  # Sample unsupervised dataset for KMeans/PCA
│  └─ processed/                  # Cleaned/transformed datasets
│     └─ README.md                # Best practices for storing processed data
├─ models/                        # Saved model artifacts and persistence
│  ├─ README.md                   # Notes about trained model files and usage
│  └─ model_persistence.py        # ModelRegistry class for save/load/list operations
├─ notebooks/                     # Experiment notebooks for exploration
│  ├─ README.md                   # Notebook conventions and best practices
│  └─ exploration.ipynb           # End-to-end EDA, training, and model comparison
└─ tests/                         # Automated test suite (unit + integration tests)
   ├─ README.md                   # Testing scope and structure guidance
   ├─ conftest.py                 # Pytest fixtures and shared test configuration
   ├─ test_services.py            # Unit tests for ML services (5 tests)
   └─ test_routes.py              # Integration tests for API endpoints (11 tests)
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

## Running Tests

The project includes comprehensive automated tests covering both services and API endpoints.

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
# Unit tests for ML services
pytest tests/test_services.py -v

# Integration tests for API endpoints
pytest tests/test_routes.py -v
```

### Run with Coverage Report

```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Manual API Endpoint Testing

For manual testing of all endpoints without pytest:

```bash
python test_api_endpoints.py
```

This script tests all 9 major endpoints and displays detailed results.

**Test Results (Current):**
- Total Tests: 16
- Passed: 16 ✅
- Failed: 0
- Execution Time: ~1.7 seconds

## Interactive API Testing with Swagger UI

FastAPI automatically generates an interactive API documentation interface called **Swagger UI**. You can test all endpoints directly from your browser without writing any code.

### How to Access Swagger UI

Once the server is running, open your browser and navigate to:

```
http://127.0.0.1:8000/docs
```

### Swagger UI Interface Preview

![FastAPI Swagger UI](./app/FastAPI%20-%20Swagger%20UI.png)

### Understanding Each Part

1. **Endpoint List (Left Side)** - All available API routes organized by category
   - Green `GET` - Retrieve data (read-only)
   - Dark Blue `POST` - Submit data (create/process)

2. **Endpoint Details (Center)** - Click any endpoint to expand and see:
   - **Summary** - Brief description of what the endpoint does
   - **Parameters** - Input fields (file, target_column, etc.)
   - **Request Body** - Data format required
   - **Try it Out** - Button to test the endpoint interactively

3. **Testing an Endpoint**
   - Click **"Try it out"** button
   - Fill in required parameters (file upload, column names, etc.)
   - Click **"Execute"** to send the request
   - View the response code, headers, and JSON output below

4. **Response Section (Bottom)** - Shows:
   - **Code** - HTTP status (200 = success, 400 = bad request, 500 = server error)
   - **Response Body** - JSON output with results, metrics, or error details
   - **Response Headers** - Content-Type, timestamp, etc.

### Example Workflow in Swagger UI

1. Select `/train-classification-logistic` endpoint
2. Click **"Try it out"**
3. Choose `sample_classification.csv` from `data/raw/`
4. Enter `target` in the `target_column` field
5. Click **"Execute"**
6. View training results including accuracy, precision, recall, and confusion matrix

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

## Sample Test Datasets

Use the ready synthetic datasets in `data/raw/` to test endpoints quickly:

- `data/raw/sample_regression.csv`
	- Use with `POST /upload` (contains `price` target column).
- `data/raw/sample_classification.csv`
	- Use with classification endpoints and set `target_column=target`.
- `data/raw/sample_unsupervised.csv`
	- Use with `POST /train-clustering-kmeans` and `POST /train-pca`.

These files are committed in the repository, so anyone can download and upload them directly via Swagger UI (`/docs`) or cURL.

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

## Extended Features

### 1. Jupyter Notebook Exploration (`notebooks/exploration.ipynb`)

A comprehensive Jupyter notebook that demonstrates:

- **Data Loading & EDA** - Load datasets and explore distributions
- **Statistical Analysis** - Summary statistics, missing values, target balance
- **Feature Analysis** - Correlation matrices, feature distributions by class
- **Model Training** - Train Logistic Regression and Random Forest side-by-side
- **Model Evaluation** - Confusion matrices, classification reports, feature importance
- **Visualization** - Histograms, heatmaps, bar charts, comparison plots
- **Overfitting Detection** - Train/test accuracy gap analysis

**Run the notebook:**
```bash
jupyter notebook notebooks/exploration.ipynb
```

### 2. Comprehensive Test Suite

**Unit Tests** (`tests/test_services.py`) - 5 tests:
- Test model training functions (logistic, decision tree, random forest)
- Test evaluation metrics calculation
- Validate prediction shapes and ranges

**Integration Tests** (`tests/test_routes.py`) - 11 tests:
- Health check endpoints (2 tests)
- File upload validation (3 tests)
- Classification endpoints (4 tests)
- Unsupervised learning endpoints (2 tests)

**Run all tests:**
```bash
python -m pytest tests/ -v
```

### 3. Model Persistence (`models/model_persistence.py`)

`ModelRegistry` class for saving and loading trained models:

```python
from models.model_persistence import ModelRegistry

# Initialize registry
registry = ModelRegistry(model_dir="models")

# Save a trained model with metadata
registry.save_model(
    trained_model,
    "logistic_v1",
    {"accuracy": 0.92, "date": "2024-01-15"}
)

# Load a model
loaded_model = registry.load_model("logistic_v1")

# List all saved models
all_models = registry.list_models()

# Retrieve model metadata
info = registry.get_model_info("logistic_v1")

# Delete a model
registry.delete_model("logistic_v1")
```

### 4. Pydantic Schemas (`app/models/schemas.py`)

Type-safe request/response validation with Pydantic models:

```python
from app.models.schemas import ClassificationResponse, ClusteringResponse, PCAResponse

# Automatically validates and documents API responses
# Provides IDE autocomplete and runtime type checking
```

**Schemas included:**
- `ClassificationResponse` - Logistic, Decision Tree, Random Forest, Neural Network
- `ClusteringResponse` - K-Means clustering results
- `PCAResponse` - Dimensionality reduction results
- `HealthResponse` - Health check endpoint
- `WelcomeResponse` - Welcome message

### 5. Configuration Management

**Configuration Files:**

- **`config.py`** - Centralized application settings (paths, model config, logging)
- **`pytest.ini`** - Pytest configuration for test discovery and markers
- **`.env`** - Environment variables template for local development
- **`requirements-dev.txt`** - Development dependencies (pytest, linting tools)

**Example `.env`:**
```
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
MODEL_TEST_SIZE=0.2
LOG_LEVEL=INFO
```

### 6. Test Fixtures (`tests/conftest.py`)

Pytest fixtures for creating sample datasets:

```python
# Available fixtures:
# - sample_classification_data: Classification dataset with train/test split
# - sample_regression_data: Regression dataset for testing models
# - sample_unsupervised_data: Clustering dataset with artificial clusters
```

### 7. API Testing Script (`test_api_endpoints.py`)

Standalone Python script for manual testing without pytest:

```bash
python test_api_endpoints.py
```

Tests all endpoints with real data and displays:
- HTTP status codes
- Model names and accuracy scores
- Response structure validation
- Execution time

## Future Improvements

- Add model persistence and loading (joblib/pickle)
- Add cross-validation and hyperparameter search
- Add multi-class and imbalanced-dataset support
- Add authentication and rate limiting for APIs
- Add experiment tracking (MLflow or Weights & Biases)
- Add Docker and CI/CD pipelines
- Expand test coverage with integration and contract tests
- Add frontend dashboard for dataset upload and result visualization

## Project Status & Completion

**✅ PROJECT FINALIZED - PRODUCTION READY**

This project has been thoroughly developed, tested, and documented:

- **16/16 Tests Passing** ✓ (100% success rate)
- **9 API Endpoints** - All functional and tested
- **Comprehensive Documentation** - README files for every module
- **Professional Code Organization** - Clean 3-tier architecture
- **Type Safety** - Pydantic schemas for all responses
- **Test Coverage** - Unit and integration tests included
- **Configuration Management** - Environment-based settings
- **Development Tools** - Jupyter notebooks, fixtures, API scripts

For detailed completion information, see [PROJECT_FINALIZATION.md](PROJECT_FINALIZATION.md).

## Author

Visura Rodrigo
