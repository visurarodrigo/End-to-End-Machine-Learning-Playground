# App - FastAPI Application

This folder contains the core FastAPI application organized using clean architecture.

## Structure

### `main.py`
FastAPI entry point that assembles routers from `routes/`.

### `routes/`
API route handlers organized by feature:
- `core.py` - welcome, health, and upload/regression workflow
- `classification.py` - classification endpoints
- `unsupervised.py` - KMeans clustering and PCA

### `services/`
Reusable ML logic: model training and evaluation (independent of HTTP).

### `utils/`
Shared utility helpers and validation functions, including CSV upload parsing.

### `models/`
Pydantic schemas for request/response validation and auto-documentation.
