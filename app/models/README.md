# App Models (Pydantic Schemas)

This folder contains **Pydantic schemas** for request/response validation in FastAPI endpoints.

## Purpose

Pydantic schemas provide:

- **Type Validation** - Automatic validation of incoming/outgoing data
- **Documentation** - Auto-generated OpenAPI/Swagger docs with field descriptions
- **IDE Support** - Autocomplete and type hints in your editor
- **API Contracts** - Clear, enforceable contracts between client and server

## Schemas in `schemas.py`

### Classification Responses
- `ClassificationResponse` - Accuracy, precision, recall, F1, confusion matrix

### Unsupervised Learning Responses
- `ClusteringResponse` - K-Means cluster labels and centers
- `PCAResponse` - Transformed rows and explained variance

### Utility Responses
- `HealthResponse` - API health status
- `WelcomeResponse` - Welcome message

## Usage Example

```python
from app.models.schemas import ClassificationResponse

@app.post("/train-model", response_model=ClassificationResponse)
async def train_model(...) -> ClassificationResponse:
    # Response automatically validated  
    return ClassificationResponse(...)
```

## Adding New Schemas

1. Define a `BaseModel` class in `schemas.py`
2. Use `Field()` for documentation
3. Add full type hints
4. Use in endpoint with `response_model` parameter

Example:
```python
from pydantic import BaseModel, Field

class MyResponse(BaseModel):
    status: str = Field(..., description="Status message")
    data: dict = Field(..., description="Response data")
```

