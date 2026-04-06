# End-to-End Machine Learning Playground

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-16%2F16-brightgreen.svg)]()

This repository contains a FastAPI backend and a Streamlit frontend for common machine learning workflows from CSV data. It supports regression analysis, classification, KMeans clustering, and PCA-based dimensionality reduction through both a web UI and REST API.

## Project Layout

- [app/README.md](app/README.md): FastAPI backend structure, routes, services, and utilities.
- [streamlit_app/README.md](streamlit_app/README.md): frontend pages and UI workflow.
- [tests/README.md](tests/README.md): automated test coverage.
- [data/README.md](data/README.md): bundled sample datasets.
- [notebooks/README.md](notebooks/README.md): exploratory notebook guidance.

## What It Does

- Accepts CSV uploads and previews data.
- Runs regression workflows with model comparison and evaluation metrics.
- Trains classification models and reports accuracy and confusion matrix metrics.
- Runs KMeans clustering and PCA on numeric data.
- Provides an interactive Streamlit interface.

## Quick Start

### Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

### Frontend

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

App URL: http://127.0.0.1:8501

## API Summary

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | / | Welcome response |
| GET | /health | Backend health check |
| POST | /upload | Regression workflow for uploaded CSV files |
| POST | /train-classification-logistic | Logistic regression |
| POST | /train-classification-decision-tree | Decision tree classification |
| POST | /train-classification-random-forest | Random forest classification |
| POST | /train-classification-neural-network | Neural network classification |
| POST | /train-clustering-kmeans | KMeans clustering |
| POST | /train-pca | PCA transformation |

## Project Structure

```text
End to End Machine Learning Playground/
|
|-- README.md
|-- requirements.txt
|-- requirements-dev.txt
|-- config.py
|-- pytest.ini
|-- LICENSE
|-- PROJECT_FINALIZATION.md
|
|-- app/
|   |-- main.py
|   |-- routes/
|   |   |-- core.py
|   |   |-- classification.py
|   |   |-- unsupervised.py
|   |-- services/
|   |   |-- model_service.py
|   |   |-- evaluation_service.py
|   |-- models/
|   |   |-- schemas.py
|   |-- utils/
|       |-- csv_utils.py
|
|-- streamlit_app/
|   |-- app.py
|   |-- requirements.txt
|   |-- pages/
|   |   |-- 1_Upload.py
|   |   |-- 2_Regression.py
|   |   |-- 3_Classification.py
|   |   |-- 4_Unsupervised.py
|   |-- utils/
|   |   |-- api_client.py
|   |   |-- sample_data.py
|   |-- Screen Shots/
|
|-- data/
|   |-- raw/
|   |   |-- sample_regression.csv
|   |   |-- sample_classification.csv
|   |   |-- sample_unsupervised.csv
|   |-- processed/
|
|-- models/
|   |-- model_persistence.py
|
|-- notebooks/
|   |-- exploration.ipynb
|
|-- tests/
|   |-- conftest.py
|   |-- test_services.py
|   |-- test_routes.py
|
|-- scripts/
    |-- manual_api_smoke_test.py
    |-- manual_feature_check.py
```

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | / | Welcome message |
| GET | /health | Health check |

### Data and Regression

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | /upload | Upload CSV and run regression workflow |

### Classification Endpoints

| Method | Endpoint | Parameters |
| --- | --- | --- |
| POST | /train-classification-logistic | file, target_column |
| POST | /train-classification-decision-tree | file, target_column, max_depth |
| POST | /train-classification-random-forest | file, target_column |
| POST | /train-classification-neural-network | file, target_column |

### Unsupervised Endpoints

| Method | Endpoint | Parameters |
| --- | --- | --- |
| POST | /train-clustering-kmeans | file, k |
| POST | /train-pca | file, n_components |

### Example: Logistic Regression via cURL

```bash
curl -X POST "http://127.0.0.1:8000/train-classification-logistic" \\
  -F "file=@data/raw/sample_classification.csv" \\
  -F "target_column=target"
```

## Streamlit Frontend

### Home Page

![Home Page](./streamlit_app/Screen%20Shots/home%20page.png)

### Upload and Sample Data

![Upload Page](./streamlit_app/Screen%20Shots/upload%20csv%20page.png)

### Regression Dashboard

![Regression Page](./streamlit_app/Screen%20Shots/Regression%20page.png)

### Classification Studio

![Classification Page](./streamlit_app/Screen%20Shots/Classification%20page.png)

### Unsupervised Explorer

![Unsupervised Page](./streamlit_app/Screen%20Shots/Unsupervised%20page.png)

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Coverage Report

```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Run Specific Test Files

```bash
pytest tests/test_services.py -v
pytest tests/test_routes.py -v
```

### Manual Scripts

```bash
python scripts/manual_api_smoke_test.py
python scripts/manual_feature_check.py
```

## Interactive API Testing

1. Start backend: `uvicorn app.main:app --reload`
2. Open docs: http://127.0.0.1:8000/docs

![FastAPI Swagger UI](./app/FastAPI%20-%20Swagger%20UI.png)

## Extended Features

### Notebook

- Notebook path: [notebooks/exploration.ipynb](notebooks/exploration.ipynb)
- Includes EDA, model comparison, metrics, and visualization.

Run it:

```bash
jupyter notebook notebooks/exploration.ipynb
```

### Model Persistence

```python
from models.model_persistence import ModelRegistry

registry = ModelRegistry(model_dir="models")

registry.save_model(trained_model, "logistic_v1", {"accuracy": 0.92})
model = registry.load_model("logistic_v1")
all_models = registry.list_models()
registry.delete_model("logistic_v1")
```

### Configuration

- `config.py`: centralized settings
- `pytest.ini`: test configuration
- `.env`: local environment variables

Example:

```env
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
MODEL_TEST_SIZE=0.2
LOG_LEVEL=INFO
```

## Troubleshooting

### Port 8000 already in use

```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
uvicorn app.main:app --port 8001
```

### Module import errors

```bash
pip install -r requirements.txt
```

### Streamlit cannot reach backend

```bash
uvicorn app.main:app --reload
curl http://127.0.0.1:8000/health
```

## Documentation

| File | Purpose |
| --- | --- |
| [app/README.md](./app/README.md) | Backend architecture and module responsibilities |
| [app/routes/README.md](./app/routes/README.md) | API endpoint organization |
| [app/services/README.md](./app/services/README.md) | ML service layer design |
| [app/models/README.md](./app/models/README.md) | Pydantic schemas and types |
| [data/README.md](./data/README.md) | Dataset organization and data additions |
| [tests/README.md](./tests/README.md) | Testing scope and structure |
| [notebooks/README.md](./notebooks/README.md) | Notebook conventions |
| [streamlit_app/README.md](./streamlit_app/README.md) | Frontend documentation and screenshots |
| [PROJECT_FINALIZATION.md](./PROJECT_FINALIZATION.md) | Project completion summary |

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

## Author

Visura Rodrigo

LinkedIn: [linkedin.com/in/visura-rodrigo-6aa98527a](https://www.linkedin.com/in/visura-rodrigo-6aa98527a)

Last updated: April 2026
