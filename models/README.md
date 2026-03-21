# Models - Trained Model Artifacts & Persistence

This folder stores trained machine learning models and their metadata.

## Files

### `model_persistence.py`
Contains `ModelRegistry` class for saving, loading, and managing trained models.

**Key Methods:**
- `save_model(model, name, metadata)` - Save model to disk with optional metadata
- `load_model(name)` - Load a saved model by name
- `list_models()` - List all saved models
- `delete_model(name)` - Delete a model and its metadata
- `get_model_info(name)` - Retrieve model metadata

## Usage Example

```python
from models.model_persistence import ModelRegistry

# Initialize registry
registry = ModelRegistry(model_dir="models")

# Save a trained model
registry.save_model(
    trained_model,
    "logistic_v1",
    {"accuracy": 0.92, "date": "2024-01-15", "dataset": "sample_classification.csv"}
)

# Load a previously trained model
model = registry.load_model("logistic_v1")
predictions = model.predict(X_test)

# View all saved models
models_list = registry.list_models()
print(models_list)  # Output: ['logistic_v1', 'random_forest_v1']

# Get model information
info = registry.get_model_info("logistic_v1")
print(info)  # Output: {'accuracy': '0.92', 'date': '2024-01-15', ...}

# Delete a model
registry.delete_model("logistic_v1")
```

## Directory Structure

```
models/
├── README.md                      # This file
├── model_persistence.py           # Registry class for model management
├── logistic_v1.pkl               # Saved model (binary)
├── logistic_v1_meta.txt          # Model metadata (human-readable)
├── random_forest_v1.pkl          # Another saved model
└── random_forest_v1_meta.txt     # Its metadata
```

## Saving Models

Models are saved in **pickle format** (.pkl) using joblib:

```python
# Automatic via ModelRegistry
registry.save_model(model, "my_model", {"accuracy": 0.88})

# Manual saving (not recommended)
import joblib
joblib.dump(model, "models/my_model.pkl")
```

## Managing Metadata

Metadata is stored as **human-readable text files** (.txt):

```
accuracy: 0.92
date: 2024-01-15
dataset: sample_classification.csv
training_time: 2.5s
features_used: 4
```

Retrieve metadata:
```python
info = registry.get_model_info("logistic_v1")
print(f"Model accuracy: {info['accuracy']}")
```

## Best Practices

- **Version Models** - Use naming like v1, v2, v3 for model versions
- **Store Metadata** - Always save accuracy, date, dataset info
- **Clean Up** - Delete old/worse-performing models periodically
- **Don't Commit Large Files** - Add `.pkl` files to `.gitignore`
- **Document Hyperparameters** - Include training parameters in metadata

## Integration with API

Save trained models from API endpoints:

```python
from models.model_persistence import ModelRegistry

@app.post("/train-classification-logistic")
async def train_classification_logistic(...):
    # ... training code ...
    
    registry = ModelRegistry()
    registry.save_model(
        trained_model,
        f"logistic_{datetime.now().strftime('%Y%m%d')}",
        {
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "date": datetime.now().isoformat(),
            "dataset": file.filename
        }
    )
    
    return response
```
