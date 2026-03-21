# Routes - API Endpoint Definitions

This folder contains FastAPI route handlers organized by feature. Routes stay thin by delegating logic to services.

## Modules

### `classification.py`

Classification model training endpoints:
- `POST /train-classification-logistic` - Logistic Regression with scaling
- `POST /train-classification-decision-tree` - Decision Tree (supports max_depth)
- `POST /train-classification-random-forest` - Random Forest ensemble
- `POST /train-classification-neural-network` - Neural Network (TensorFlow optional)

**Common Parameters:**
- `file` - CSV file with training data
- `target_column` - Name of target variable column
- `max_depth` (optional) - Decision tree max depth

**Response:** ClassificationResponse with accuracy, precision, recall, F1, confusion matrix

## Design Pattern

Keep routes thin:
1. Validate input parameters
2. Delegate ML logic to services
3. Format responses using Pydantic schemas
4. Handle HTTP concerns (status codes, errors)
