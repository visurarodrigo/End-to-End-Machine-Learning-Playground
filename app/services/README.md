# Services - Business & ML Logic

Reusable, stateless service functions for ML operations. Services are independent of HTTP and can be used in routes, tests, and notebooks.

## Modules

### `model_service.py`

Model training helpers:
- `train_logistic_classifier(X_train, y_train, X_test)` 
  - Logistic Regression with StandardScaler pipeline
- `train_decision_tree_classifier(X_train, y_train, X_test, max_depth)`
  - Decision Tree with optional depth control
- `train_random_forest_classifier(X_train, y_train, X_test)`
  - Random Forest (100 estimators)

Returns: `(y_train_pred, y_test_pred)` - Predictions on both sets for overfitting detection

### `evaluation_service.py`

Evaluation metrics:
- `calculate_train_test_accuracy()` - Accuracy and overfitting gap
- `calculate_classification_metrics()` - Precision, recall, F1, confusion matrix
- `build_classification_metrics_response()` - Combined metrics response

## Key Principle

Services are pure functions: same inputs → same outputs, no side effects. Can be imported directly by tests and notebooks.
