from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile


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
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
