"""Application configuration settings."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TESTS_DIR = BASE_DIR / "tests"

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# Model Configuration
MODEL_TEST_SIZE = float(os.getenv("MODEL_TEST_SIZE", "0.2"))
MODEL_RANDOM_STATE = int(os.getenv("MODEL_RANDOM_STATE", "42"))
MODEL_SAVE_DIR = MODELS_DIR / "saved_models"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Feature Scaling
USE_STANDARD_SCALER = os.getenv("USE_STANDARD_SCALER", "true").lower() == "true"

# Create directories if they don't exist
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
