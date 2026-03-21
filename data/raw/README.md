# Raw Data - Synthetic Sample Datasets

Read-only sample datasets for quick API testing. All files are synthetic (randomly generated) for demonstration purposes.

## Datasets

### `sample_classification.csv`
**Use With:** `/train-classification-*` endpoints
**Columns:** age, income, credit_score, loan_amount, tenure_months, target (0/1)
**Rows:** ~100 samples
**Type:** Binary classification (loan approval prediction)

### `sample_regression.csv`
**Use With:** `/upload` endpoint (regression workflow)
**Columns:** age, income, credit_score, loan_amount, tenure_months, price
**Rows:** ~100 samples
**Type:** Regression (price/value prediction with model comparison)

### `sample_unsupervised.csv`
**Use With:** `/train-clustering-kmeans`, `/train-pca`
**Columns:** feature_1 through feature_5 (numeric only, no target column)
**Rows:** ~150 samples
**Type:** Unsupervised learning (clustering and dimensionality reduction)

## Quick Test via Swagger UI

1. **Start API:**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Open Swagger UI:**
   ```
   http://127.0.0.1:8000/docs
   ```

3. **Test Endpoint:**
   - Select any classification endpoint
   - Click "Try it out"
   - Upload a CSV file
   - Set target_column (e.g., "target")
   - Click "Execute"
   - View results with model metrics

## Important Notes

⚠️ **Synthetic Data Only** - For testing and learning purposes only.
