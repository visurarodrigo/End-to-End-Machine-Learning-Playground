from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


app = FastAPI()


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
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = float(mean_squared_error(y_test, y_pred))
    
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
        "mse": mse,
        "predictions_sample": predictions_sample,
        "prediction_analysis": prediction_analysis,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
