"""Shared helpers for validating and reading uploaded CSV files."""

from io import BytesIO

import pandas as pd
from fastapi import HTTPException, UploadFile


async def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Validate an uploaded file and parse it as a CSV DataFrame."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        file_bytes = await file.read()
        return pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {exc}") from exc
