from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier


app = FastAPI()


def _build_classification_metrics_response(
    *,
    message: str,
    model_name: str,
    y_test: pd.Series,
    y_pred: object,
) -> dict[str, object]:
    """Build a consistent response payload for binary classification endpoints."""
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "message": message,
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "actual_values": y_test.iloc[:10].tolist(),
        "predicted_values": y_pred[:10].tolist(),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the End-to-End ML Playground"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", summary="Upload a CSV file")
async def upload_file(file: UploadFile = File(...)) -> dict[str, object]:
    """Accept a CSV file upload and return basic file metadata."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        file_bytes = await file.read()
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {exc}") from exc

    rows, columns = df.shape
    
    # Calculate missing values
    missing_per_column = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    # Create a cleaned copy with imputation
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype in ["float64", "int64"]:
                # Fill numeric columns with mean
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            else:
                # Fill non-numeric columns with mode (most frequent value)
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)
    
    # Determine if cleaning was performed
    has_missing = total_missing > 0
    cleaning_message = (
        "Missing values detected and cleaned DataFrame created with mean imputation for numeric columns and mode imputation for non-numeric columns."
        if has_missing
        else "No missing values detected in the dataset."
    )
    
    # Train-test split preparation
    target_column = "price"  # Hardcoded for now, can be parameterized later
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}"
        )
    
    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Collect split information
    split_info = {
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
    }
    
    # Train baseline Linear Regression model (without feature scaling)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on baseline test set
    y_pred = model.predict(X_test)

    # Calculate baseline Mean Squared Error
    original_mse = float(mean_squared_error(y_test, y_pred))

    # Fit scaler only on training features, then transform train and test features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a new Linear Regression model on scaled features
    scaled_model = LinearRegression()
    scaled_model.fit(X_train_scaled, y_train)

    # Predict and evaluate using scaled test features
    y_pred_scaled = scaled_model.predict(X_test_scaled)
    scaled_mse = float(mean_squared_error(y_test, y_pred_scaled))

    # Build polynomial features from training data and apply the same transform to test data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale polynomial features for stable optimization
    poly_scaler = StandardScaler()
    X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = poly_scaler.transform(X_test_poly)

    # Train and evaluate polynomial regression model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly_scaled, y_train)
    y_pred_poly = poly_model.predict(X_test_poly_scaled)
    polynomial_mse = float(mean_squared_error(y_test, y_pred_poly))

    # Train and evaluate regularized polynomial models
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_poly_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_poly_scaled)
    ridge_mse = float(mean_squared_error(y_test, y_pred_ridge))

    lasso_model = Lasso(alpha=0.1, max_iter=10000)
    lasso_model.fit(X_train_poly_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_poly_scaled)
    lasso_mse = float(mean_squared_error(y_test, y_pred_lasso))

    if scaled_mse < original_mse:
        scaling_message = "Scaling improved model performance (lower MSE)."
    elif scaled_mse > original_mse:
        scaling_message = "Scaling did not improve model performance (higher MSE)."
    else:
        scaling_message = "Scaling produced the same model performance (equal MSE)."

    if polynomial_mse < scaled_mse:
        polynomial_message = "Polynomial regression improved performance compared to scaled linear regression."
    elif polynomial_mse > scaled_mse:
        polynomial_message = "Polynomial regression performed worse on test data and may indicate overfitting."
    else:
        polynomial_message = "Polynomial regression produced similar test performance to scaled linear regression."

    best_regularized_name = "ridge" if ridge_mse <= lasso_mse else "lasso"
    best_regularized_mse = min(ridge_mse, lasso_mse)

    if best_regularized_mse < polynomial_mse:
        regularization_message = (
            f"Regularization improved performance versus polynomial regression alone; "
            f"{best_regularized_name.capitalize()} achieved the lowest regularized MSE and likely reduced overfitting."
        )
    elif best_regularized_mse > polynomial_mse:
        regularization_message = (
            "Regularization did not improve test performance versus polynomial regression alone and may be too strong for this dataset."
        )
    else:
        regularization_message = (
            "Regularization produced similar performance to polynomial regression alone, with potential stability benefits."
        )

    model_mse_values = {
        "original": original_mse,
        "scaled": scaled_mse,
        "polynomial": polynomial_mse,
        "ridge": ridge_mse,
        "lasso": lasso_mse,
    }
    model_comparison = {
        "original_mse": original_mse,
        "scaled_mse": scaled_mse,
        "polynomial_mse": polynomial_mse,
        "ridge_mse": ridge_mse,
        "lasso_mse": lasso_mse,
        "best_model_by_mse": min(model_mse_values, key=model_mse_values.get),
    }
    
    # Calculate residuals (prediction errors)
    residuals = y_test.values - y_pred
    
    # Collect first 5 predictions for inspection
    predictions_sample = {
        "y_test_sample": y_test.iloc[:5].tolist(),
        "y_pred_sample": y_pred[:5].tolist(),
    }
    
    # Detailed prediction analysis with residuals
    prediction_analysis = {
        "explanation": "Residuals represent the difference between actual and predicted values (actual - predicted). Smaller residuals indicate better model performance.",
        "actual_values": y_test.iloc[:5].tolist(),
        "predicted_values": y_pred[:5].tolist(),
        "residuals": residuals[:5].tolist(),
    }
    
    return {
        "filename": file.filename,
        "content_type": file.content_type or "unknown",
        "rows": int(rows),
        "columns": int(columns),
        "column_names": [str(column) for column in df.columns.tolist()],
        "missing_values": {str(k): int(v) for k, v in missing_per_column.items()},
        "total_missing_values": total_missing,
        "cleaning_status": cleaning_message,
        "preview": df.head(5).to_dict(orient="records"),
        "target_column": target_column,
        "train_test_split": split_info,
        "mse": original_mse,
        "original_mse": original_mse,
        "scaled_mse": scaled_mse,
        "polynomial_mse": polynomial_mse,
        "ridge_mse": ridge_mse,
        "lasso_mse": lasso_mse,
        "scaling_performance_message": scaling_message,
        "polynomial_performance_message": polynomial_message,
        "regularization_performance_message": regularization_message,
        "model_comparison": model_comparison,
        "predictions_sample": predictions_sample,
        "prediction_analysis": prediction_analysis,
    }


@app.post("/train-classification-logistic", summary="Train a Logistic Regression classifier")
async def train_classification_logistic(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, object]:
    """Train a Logistic Regression model from an uploaded CSV file."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        file_bytes = await file.read()
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {exc}") from exc

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

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return _build_classification_metrics_response(
        message="Logistic regression training completed successfully.",
        model_name="LogisticRegression",
        y_test=y_test,
        y_pred=y_pred,
    )


@app.post("/train-classification-decision-tree", summary="Train a Decision Tree classifier")
async def train_classification_decision_tree(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    max_depth: int | None = Form(None),
) -> dict[str, object]:
    """Train a Decision Tree Classifier model from an uploaded CSV file."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        file_bytes = await file.read()
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {exc}") from exc

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

    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    response = _build_classification_metrics_response(
        message="Decision tree training completed successfully.",
        model_name="Decision Tree Classifier",
        y_test=y_test,
        y_pred=y_pred,
    )
    response["max_depth"] = max_depth
    return response


@app.post("/train-classification-random-forest", summary="Train a Random Forest classifier")
async def train_classification_random_forest(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> dict[str, object]:
    """Train a Random Forest Classifier model from an uploaded CSV file."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        file_bytes = await file.read()
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {exc}") from exc

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

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return _build_classification_metrics_response(
        message="Random forest training completed successfully.",
        model_name="Random Forest Classifier",
        y_test=y_test,
        y_pred=y_pred,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
