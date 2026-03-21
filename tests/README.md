# Tests - Automated Test Suite

This folder contains unit tests for API routes and ML services.

## Files

### `test_services.py`
Tests for core ML services:
- **ModelService Tests** - Train logistic, decision tree, random forest classifiers
- **EvaluationService Tests** - Accuracy calculation, classification metrics, confusion matrices
- **Data Integrity Tests** - Verify output shapes and value ranges

## Running Tests

### With pytest (recommended)
```bash
pip install pytest
pytest tests/ -v
```

### With unittest
```bash
python -m unittest discover tests/ -v
```

### Run specific test file
```bash
python -m unittest tests.test_services -v
```

## Test Coverage

Run coverage analysis:
```bash
pip install coverage
coverage run -m pytest tests/
coverage report
```

## Adding New Tests

1. Create `test_*.py` file in this folder
2. Import unittest and test functions
3. Use `setUp()` for test data initialization
4. Use descriptive test names: `test_[function]_[scenario]`

Example:
```python
import unittest
from app.services.model_service import train_logistic_classifier

class TestLogisticClassifier(unittest.TestCase):
    def test_train_produces_predictions(self):
        # Your test here
        pass
```

## Best Practices

- Keep tests isolated and independent
- Use fixtures for common test data
- Test edge cases and error conditions
- Aim for >80% code coverage
- Run tests before committing changes
