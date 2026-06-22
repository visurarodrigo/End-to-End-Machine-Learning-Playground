"""Regression service: runs the full regression workflow on a DataFrame."""

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def run_regression_workflow(df: pd.DataFrame, target_column: str = "price") -> dict:
    """Run full regression workflow and return results as a dict."""

    if target_column not in df.columns:
        return {
            "error": f"Target column '{target_column}' not found. "
                     f"Available columns: {', '.join(df.columns)}"
        }

    # --- Missing value info ---
    missing_per_column = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype in ["float64", "int64"]:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            else:
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)

    has_missing = total_missing > 0
    cleaning_message = (
        "Missing values detected and cleaned with mean/mode imputation."
        if has_missing
        else "No missing values detected."
    )

    # --- Train/test split ---
    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Models ---
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

    # --- Messages ---
    scaling_message = (
        "Scaling improved model performance (lower MSE)." if scaled_mse < original_mse
        else "Scaling did not improve model performance (higher MSE)." if scaled_mse > original_mse
        else "Scaling produced the same model performance."
    )
    polynomial_message = (
        "Polynomial regression improved performance vs scaled linear regression." if polynomial_mse < scaled_mse
        else "Polynomial regression performed worse — may indicate overfitting." if polynomial_mse > scaled_mse
        else "Polynomial regression produced similar performance."
    )
    best_reg_name = "ridge" if ridge_mse <= lasso_mse else "lasso"
    best_reg_mse = min(ridge_mse, lasso_mse)
    regularization_message = (
        f"Regularization improved performance; {best_reg_name.capitalize()} achieved the lowest MSE."
        if best_reg_mse < polynomial_mse
        else "Regularization did not improve performance vs polynomial regression."
        if best_reg_mse > polynomial_mse
        else "Regularization produced similar performance to polynomial regression."
    )

    model_mse_values = {
        "original": original_mse,
        "scaled": scaled_mse,
        "polynomial": polynomial_mse,
        "ridge": ridge_mse,
        "lasso": lasso_mse,
    }
    residuals = (y_test.values - y_pred).tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": {str(k): int(v) for k, v in missing_per_column.items()},
        "total_missing_values": total_missing,
        "cleaning_status": cleaning_message,
        "preview": df.head(5).to_dict(orient="records"),
        "target_column": target_column,
        "train_test_split": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "original_mse": original_mse,
        "scaled_mse": scaled_mse,
        "polynomial_mse": polynomial_mse,
        "ridge_mse": ridge_mse,
        "lasso_mse": lasso_mse,
        "scaling_performance_message": scaling_message,
        "polynomial_performance_message": polynomial_message,
        "regularization_performance_message": regularization_message,
        "model_comparison": {
            **model_mse_values,
            "best_model_by_mse": min(model_mse_values, key=model_mse_values.get),
        },
        "prediction_analysis": {
            "explanation": "Residuals = actual − predicted. Smaller residuals = better fit.",
            "actual_values": y_test.iloc[:5].tolist(),
            "predicted_values": y_pred[:5].tolist(),
            "residuals": residuals[:5],
        },
    }