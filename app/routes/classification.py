from io import BytesIO

import pandas as pd
import tensorflow as tf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sklearn.model_selection import train_test_split

from app.services.evaluation_service import build_classification_metrics_response, calculate_train_test_accuracy
from app.services.model_service import (
    train_decision_tree_classifier,
    train_logistic_classifier,
    train_random_forest_classifier,
)


router = APIRouter(tags=["classification"])


def _read_csv_upload(file: UploadFile, file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV upload into a DataFrame with consistent error handling."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        return pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {exc}") from exc


@router.post("/train-classification-logistic", summary="Train a Logistic Regression classifier")
async def train_classification_logistic(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, object]:
    """Train a Logistic Regression model from an uploaded CSV file."""
    file_bytes = await file.read()
    df = _read_csv_upload(file, file_bytes)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}",
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    y_train_pred, y_test_pred = train_logistic_classifier(X_train, y_train, X_test)

    return build_classification_metrics_response(
        message="Logistic regression training completed successfully.",
        model_name="LogisticRegression",
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )


@router.post("/train-classification-decision-tree", summary="Train a Decision Tree classifier")
async def train_classification_decision_tree(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    max_depth: int | None = Form(None),
) -> dict[str, object]:
    """Train a Decision Tree Classifier model from an uploaded CSV file."""
    file_bytes = await file.read()
    df = _read_csv_upload(file, file_bytes)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}",
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    if max_depth is not None and max_depth <= 0:
        raise HTTPException(status_code=400, detail="max_depth must be a positive integer when provided.")

    y_train_pred, y_test_pred = train_decision_tree_classifier(X_train, y_train, X_test, max_depth=max_depth)

    response = build_classification_metrics_response(
        message="Decision tree training completed successfully.",
        model_name="Decision Tree Classifier",
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )
    response["max_depth"] = max_depth
    return response


@router.post("/train-classification-random-forest", summary="Train a Random Forest classifier")
async def train_classification_random_forest(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, object]:
    """Train a Random Forest Classifier model from an uploaded CSV file."""
    file_bytes = await file.read()
    df = _read_csv_upload(file, file_bytes)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}",
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    y_train_pred, y_test_pred = train_random_forest_classifier(X_train, y_train, X_test)

    return build_classification_metrics_response(
        message="Random forest training completed successfully.",
        model_name="Random Forest Classifier",
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )


@router.post("/train-classification-neural-network", summary="Train a Neural Network classifier")
async def train_classification_neural_network(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, object]:
    """Train a TensorFlow neural network for binary classification from an uploaded CSV file."""
    file_bytes = await file.read()
    df = _read_csv_upload(file, file_bytes)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}",
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.isnull().any():
        raise HTTPException(status_code=400, detail="Target column contains missing values. Please clean the dataset first.")

    if y.nunique(dropna=True) != 2:
        raise HTTPException(status_code=400, detail="Neural network endpoint supports binary classification only.")

    X = pd.get_dummies(X, drop_first=False)
    if X.empty:
        raise HTTPException(status_code=400, detail="No usable feature columns found after preprocessing.")

    y_encoded, class_labels = pd.factorize(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
    )

    X_train_np = X_train.to_numpy(dtype="float32")
    X_test_np = X_test.to_numpy(dtype="float32")
    y_train_np = y_train.astype("float32")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train_np.shape[1],)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(X_train_np, y_train_np, epochs=10, batch_size=32, verbose=0)

    y_train_prob = model.predict(X_train_np, verbose=0).ravel()
    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_test_prob = model.predict(X_test_np, verbose=0).ravel()
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    accuracy_payload = calculate_train_test_accuracy(y_train, y_train_pred, y_test, y_test_pred)

    return {
        "message": "Neural network training completed successfully.",
        "model": "TensorFlow Sequential Neural Network",
        **accuracy_payload,
        "actual_values": y_test[:10].tolist(),
        "predicted_values": y_test_pred[:10].tolist(),
        "predicted_probabilities": y_test_prob[:10].tolist(),
        "class_mapping": {str(index): str(label) for index, label in enumerate(class_labels.tolist())},
        "epochs": 10,
        "batch_size": 32,
    }
