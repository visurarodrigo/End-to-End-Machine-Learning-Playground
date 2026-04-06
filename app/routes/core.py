"""Core application routes for health checks and CSV upload workflows."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.utils.csv_utils import read_csv_upload


router = APIRouter(tags=["core"])


@router.get("/")
def read_root() -> dict[str, str]:
    """Return the welcome message for the API."""
    return {"message": "Welcome to the End-to-End ML Playground"}


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health status for the API."""
    return {"status": "ok"}


@router.post("/upload", summary="Upload a CSV file")
async def upload_file(file: UploadFile = File(...)) -> dict[str, object]:
    """Accept a CSV file upload and return basic file metadata and regression metrics."""
    df = await read_csv_upload(file)

    rows, columns = df.shape
    missing_per_column = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    df_cleaned = df.copy()
    for column in df_cleaned.columns:
        if df_cleaned[column].isnull().any():
            if df_cleaned[column].dtype in ["float64", "int64"]:
                df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
            else:
                mode_val = df_cleaned[column].mode()
                if not mode_val.empty:
                    df_cleaned[column].fillna(mode_val[0], inplace=True)

    has_missing = total_missing > 0
    cleaning_message = (
        "Missing values detected and cleaned DataFrame created with mean imputation for numeric columns and mode imputation for non-numeric columns."
        if has_missing
        else "No missing values detected in the dataset."
    )

    target_column = "price"
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset. Available columns: {', '.join(df.columns)}",
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    split_info = {
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
    }

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    original_mse = float(mean_squared_error(y_test, y_pred))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaled_model = LinearRegression()
    scaled_model.fit(X_train_scaled, y_train)
    y_pred_scaled = scaled_model.predict(X_test_scaled)
    scaled_mse = float(mean_squared_error(y_test, y_pred_scaled))

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_scaler = StandardScaler()
    X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = poly_scaler.transform(X_test_poly)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly_scaled, y_train)
    y_pred_poly = poly_model.predict(X_test_poly_scaled)
    polynomial_mse = float(mean_squared_error(y_test, y_pred_poly))

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
        regularization_message = "Regularization produced similar performance to polynomial regression alone, with potential stability benefits."

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

    residuals = y_test.values - y_pred
    predictions_sample = {
        "y_test_sample": y_test.iloc[:5].tolist(),
        "y_pred_sample": y_pred[:5].tolist(),
    }
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
