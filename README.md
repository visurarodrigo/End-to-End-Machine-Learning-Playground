# End-to-End Machine Learning Playground

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-16%2F16%20✓-brightgreen.svg)]()
# End-to-End Machine Learning Playground

This repository contains a FastAPI backend and a Streamlit frontend for running common machine learning workflows from CSV data. It supports regression analysis, classification, KMeans clustering, and PCA-based dimensionality reduction through both a web UI and a REST API.

## Project Layout

- [app/README.md](app/README.md) explains the FastAPI backend structure, routes, services, and utilities.
- [streamlit_app/README.md](streamlit_app/README.md) explains the frontend pages and UI workflow.
- [tests/README.md](tests/README.md) describes the automated test coverage.
- [data/README.md](data/README.md) covers the bundled sample datasets.
- [notebooks/README.md](notebooks/README.md) covers exploratory notebooks.

## What It Does

- Accepts CSV uploads and previews the data.
- Runs a regression workflow with model comparison and evaluation metrics.
- Trains classification models and reports accuracy and confusion matrix metrics.
- Runs KMeans clustering and PCA on numeric data.
- Provides a Streamlit interface for interactive use.

## Quick Start

### Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The app runs at `http://127.0.0.1:8501`.

## API Summary

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Welcome response |
| GET | `/health` | Backend health check |
| POST | `/upload` | Regression workflow for uploaded CSV files |
| POST | `/train-classification-logistic` | Logistic regression |
| POST | `/train-classification-decision-tree` | Decision tree classification |
| POST | `/train-classification-random-forest` | Random forest classification |
| POST | `/train-classification-neural-network` | Neural network classification |
| POST | `/train-clustering-kmeans` | KMeans clustering |
| POST | `/train-pca` | PCA transformation |

## Testing

Run the full test suite with:

```bash
pytest
```

Manual smoke tests are available in [scripts/manual_api_smoke_test.py](scripts/manual_api_smoke_test.py) and [scripts/manual_feature_check.py](scripts/manual_feature_check.py).

## Sample Data

Bundled datasets are located in `data/raw/` and can be used from the Streamlit app or directly against the API.

## License

See [LICENSE](LICENSE) for licensing details.
```

Now both are running! Upload a CSV or use sample datasets.

---

## Project Structure

```
End to End Machine Learning Playground/
│
├── README.md                      # This file
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development tools
├── config.py                      # App configuration
├── pytest.ini                     # Test configuration
├── LICENSE                        # MIT License
├── PROJECT_FINALIZATION.md        # Completion summary
│
├── app/                           # Backend (FastAPI)
│   ├── main.py                       # API entry point and router assembly
│   ├── routes/
│   │   ├── core.py                   # Root, health, upload workflow
│   │   ├── classification.py         # 4 classification endpoints
│   │   └── unsupervised.py           # KMeans and PCA endpoints
│   ├── services/
│   │   ├── model_service.py          # Model training logic
│   │   └── evaluation_service.py     # Metrics calculation
│   ├── models/
│   │   └── schemas.py                # Pydantic response types
│   └── utils/
│       └── csv_utils.py              # CSV upload helper
│
├── streamlit_app/                 # Frontend (Streamlit)
│   ├── app.py                        # Home page
│   ├── requirements.txt              # Streamlit dependencies
│   ├── pages/
│   │   ├── 1_Upload.py              # File upload + samples
│   │   ├── 2_Regression.py          # Regression dashboard
│   │   ├── 3_Classification.py      # Classification studio
│   │   └── 4_Unsupervised.py        # Clustering & PCA
│   ├── utils/
│   │   ├── api_client.py            # Backend communication
│   │   └── sample_data.py           # Dataset loaders
│   └── Screen Shots/                 # Preview images
│
├── data/                          # Datasets
│   ├── raw/
│   │   ├── sample_regression.csv    # 500 rows × 6 features
│   │   ├── sample_classification.csv# 500 rows × 6 features
│   │   └── sample_unsupervised.csv  # 500 rows × 4 features
│   └── processed/                   # (for future use)
│
├── models/                        # Saved ML models
│   └── model_persistence.py          # Save/load models
│
├── notebooks/                     # Jupyter exploration
│   └── exploration.ipynb             # EDA & model comparison
│
├── tests/                         # Test suite
│   ├── conftest.py                   # Test fixtures
│   ├── test_services.py              # Unit tests (5)
│   └── test_routes.py                # Integration tests (11)

└── scripts/                       # Manual smoke checks
    ├── manual_api_smoke_test.py      # API smoke test against a running server
    └── manual_feature_check.py       # Sample-data feature check
```

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |

### Data + Regression

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload CSV, run regression workflow |

### Classification (4 endpoints)

| Method | Endpoint | Parameters |
|--------|----------|------------|
| `POST` | `/train-classification-logistic` | file, target_column |
| `POST` | `/train-classification-decision-tree` | file, target_column, max_depth |
| `POST` | `/train-classification-random-forest` | file, target_column |
| `POST` | `/train-classification-neural-network` | file, target_column |

### Unsupervised Learning

| Method | Endpoint | Parameters |
|--------|----------|------------|
| `POST` | `/train-clustering-kmeans` | file, k |
| `POST` | `/train-pca` | file, n_components |

### Interactive Testing

Once the backend is running, open **Swagger UI**:
```
http://127.0.0.1:8000/docs
```

Click "Try it out" on any endpoint, fill parameters, and execute!

### Example: Logistic Regression via cURL

```bash
curl -X POST "http://127.0.0.1:8000/train-classification-logistic" \
  -F "file=@data/raw/sample_classification.csv" \
  -F "target_column=target"
```

**Response (JSON):**
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
  "confusion_matrix": {
    "true_negatives": 42,
    "false_positives": 6,
    "false_negatives": 5,
    "true_positives": 47
  }
}
```

---

## Streamlit Interactive Frontend

### Home Page

![Home Page](./streamlit_app/Screen%20Shots/home%20page.png)

**Overview:** Welcome page with quick-start guide, feature cards, and API health status.

### Upload & Sample Data

![Upload Page](./streamlit_app/Screen%20Shots/upload%20csv%20page.png)

**Features:**
- Drag-and-drop CSV upload
- 3 pre-built sample datasets (one click)
- Instant data preview
- Schema validation

### Regression Dashboard

![Regression Page](./streamlit_app/Screen%20Shots/Regression%20page.png)

**Features:**
- Train 5 regression models simultaneously
- MSE comparison bar chart
- Best model highlight
- Actual vs Predicted scatter plot

### Classification Studio

![Classification Page](./streamlit_app/Screen%20Shots/Classification%20page.png)

**Features:**
- 4 classification algorithms
- Multi-model accuracy comparison
- Precision, Recall, F1 metrics
- Confusion matrix heatmaps
- Overfitting detection (accuracy gap)

### Unsupervised Explorer

![Unsupervised Page](./streamlit_app/Screen%20Shots/Unsupervised%20page.png)

**Features:**
- K-Means clustering with adjustable k
- PCA with explained variance
- Cluster distribution chart
- Component importance visualization

### Running the Frontend

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

For detailed feature explanations, see [streamlit_app/README.md](./streamlit_app/README.md).

---

## Testing

### Run All Tests

```bash
pytest tests/ -v

# Output:
# tests/test_services.py::test_logistic_training ✓
# tests/test_routes.py::test_health_check ✓
# tests/test_routes.py::test_upload ✓
# ... (16 tests total)
```

### Test Coverage

```bash
pytest tests/ -v --cov=app --cov-report=html

# Opens: htmlcov/index.html
```

### Run Specific Test File

```bash
# Unit tests
pytest tests/test_services.py -v

# Integration tests
pytest tests/test_routes.py -v
```

### Manual Testing Script

```bash
python test_api_endpoints.py

# Tests all 9 endpoints with real data
```

### Current Test Results

```
Total: 16 tests
Passed: 16 (100%)
Failed: 0
Execution: ~1.7 seconds
```

**Coverage:**
- Unit tests: ML services, model training, evaluation metrics
- Integration tests: API endpoints, data validation, response schemas

---

## Interactive API Testing (Swagger UI)

### How to Access

1. Start the backend: `uvicorn app.main:app --reload`
2. Open browser: `http://127.0.0.1:8000/docs`

### Using Swagger UI

![FastAPI Swagger UI](./app/FastAPI%20-%20Swagger%20UI.png)

**Step-by-step:**
1. Click any endpoint to expand
2. Click "Try it out"
3. Fill in parameters (file upload, column names)
4. Click "Execute"
5. View response below

**Example Workflow:**
1. Select `/train-classification-logistic`
2. Upload: `data/raw/sample_classification.csv`
3. Set `target_column=target`
4. Click Execute → See accuracy, precision, recall, confusion matrix

---

## Extended Features

### Jupyter Notebook (`notebooks/exploration.ipynb`)

Comprehensive ML exploration with:
- **EDA** - Data loading, distributions, missing values
- **Statistical Analysis** - Summary stats, target balance
- **Model Training** - Logistic Regression vs Random Forest
- **Evaluation** - Confusion matrices, feature importance
- **Visualizations** - Histograms, heatmaps, comparisons
- **Overfitting Detection** - Train/test gap analysis

**Run it:**
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Model Persistence

Save and load trained models:

```python
from models.model_persistence import ModelRegistry

registry = ModelRegistry(model_dir="models")

# Save
registry.save_model(trained_model, "logistic_v1", 
                   {"accuracy": 0.92, "date": "2024-01-15"})

# Load
model = registry.load_model("logistic_v1")

# List all
all_models = registry.list_models()

# Delete
registry.delete_model("logistic_v1")
```

### Pydantic Schemas

Type-safe API responses:

```python
from app.models.schemas import ClassificationResponse

# Automatic validation, IDE autocomplete, runtime type checking
```

**Included:**
- `ClassificationResponse` - Classification results
- `ClusteringResponse` - K-Means results
- `PCAResponse` - PCA results
- `HealthResponse` - Health check
- `WelcomeResponse` - Welcome message

### Configuration Management

**Files:**
- `config.py` - Centralized settings
- `pytest.ini` - Test configuration
- `.env` - Environment variables (local development)

**Example `.env`:**
```
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
MODEL_TEST_SIZE=0.2
LOG_LEVEL=INFO
```

---

## Troubleshooting

### "Port 8000 already in use"

```bash
# Find process using port 8000 (Windows)
netstat -ano | findstr :8000

# Kill it (replace PID with actual number)
taskkill /PID <PID> /F

# Or use different port
uvicorn app.main:app --port 8001
```

### "ModuleNotFoundError: No module named 'app'"

```bash
# Ensure you're in the project root directory
cd "End to End Machine Learning Playground"

# Reinstall dependencies
pip install -r requirements.txt
```

### "Streamlit can't find FastAPI backend"

```bash
# Ensure backend is running
uvicorn app.main:app --reload

# Check it's accessible
curl http://127.0.0.1:8000/health
```

### CSV Upload Fails

- Ensure CSV has at least 20 rows (ML needs training data)
- Check columns are numeric (except target column)
- Verify target column exists and has binary/multiclass labels

### Import Errors with TensorFlow (Neural Network)

```bash
# Install TensorFlow if not already present
pip install tensorflow

# Verify
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## 🤝 Contributing

Found a bug? Have a feature idea?

1. **Create an issue** with detailed description
2. **Fork the repo** and create a feature branch
3. **Make your changes** with test coverage
4. **Submit a pull request** with clear title + description

---

## 📚 Sample Test Datasets

All in `data/raw/`:

| Dataset | Rows | Columns | Use Case |
|---------|------|---------|----------|
| `sample_regression.csv` | 500 | 6 (numeric + price target) | Regression models |
| `sample_classification.csv` | 500 | 6 (mixed + binary target) | Classification models |
| `sample_unsupervised.csv` | 500 | 4 (numeric only) | Clustering & PCA |

Load directly via:
- Streamlit **Upload** page (one-click buttons)
- Swagger UI endpoint testing
- Python script with `requests` library

---

## 🎓 Learning Outcomes

Build this project to master:

- **API Design** - RESTful endpoints, request/response handling
- **ML Pipelines** - Data prep, training, evaluation workflows
- **Model Comparison** - Metrics, train/test splits, overfitting detection
- **Code Architecture** - 3-tier design, separation of concerns
- **Type Safety** - Pydantic schemas, IDE autocomplete
- **Testing** - Unit tests, integration tests, fixtures
- **Frontend Development** - Building data apps with Streamlit
- **Full-Stack ML** - Backend + frontend integration

---

## 📄 Project Documentation

Each module includes detailed documentation:

| File | Purpose |
|------|---------|
| [app/README.md](./app/README.md) | Backend architecture & module responsibilities |
| [app/routes/README.md](./app/routes/README.md) | API endpoint organization |
| [app/services/README.md](./app/services/README.md) | ML service layer design |
| [app/models/README.md](./app/models/README.md) | Pydantic schemas & types |
| [data/README.md](./data/README.md) | Dataset organization & adding new data |
| [tests/README.md](./tests/README.md) | Testing scope & structure |
| [notebooks/README.md](./notebooks/README.md) | Jupyter notebook conventions |
| [streamlit_app/README.md](./streamlit_app/README.md) | Frontend documentation & screenshots |
| [PROJECT_FINALIZATION.md](./PROJECT_FINALIZATION.md) | Project completion summary |

---

## 👤 Author

**Visura Rodrigo**

🔗 **LinkedIn:** [linkedin.com/in/visura-rodrigo-6aa98527a](https://www.linkedin.com/in/visura-rodrigo-6aa98527a)

---

## 📋 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

**Made with ❤️ by Visura Rodrigo**

*Last Updated: March 2026*
