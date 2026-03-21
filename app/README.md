# App - FastAPI Application

This folder contains the core FastAPI application organized using clean architecture.

## Structure

### `main.py`
FastAPI entry point with core endpoints:
- GET / - Welcome message
- GET /health - Health check
- POST /upload - CSV upload with regression
- POST /train-clustering-kmeans - K-Means clustering
- POST /train-pca - PCA dimensionality reduction

### `routes/`
API route handlers organized by feature (classification endpoints).

### `services/`
Reusable ML logic: model training and evaluation (independent of HTTP).

### `utils/`
Shared utility helpers and validation functions.

### `models/`
Pydantic schemas for request/response validation and auto-documentation.
