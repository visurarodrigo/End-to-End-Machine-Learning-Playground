# Processed Data - Cleaned & Transformed Datasets

Storage location for datasets after cleaning, transformation, and feature engineering. These datasets are production-ready for model training and evaluation.

## Folder Organization

```
processed/
├── classification/
│   ├── train_features.csv
│   ├── train_labels.csv
│   ├── test_features.csv
│   └── test_labels.csv
├── regression/
│   ├── train_features.csv
│   ├── train_labels.csv
│   ├── test_features.csv
│   └── test_labels.csv
└── metadata/
    ├── preprocessing_steps.json
    ├── feature_definitions.json
    └── data_statistics.json
```

## Best Practices

### 1. **Separate Features & Labels**
```python
# ✅ Good practice
train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')

# ❌ Avoid mixing
train_full = pd.read_csv('train_full.csv')
```

### 2. **Document Preprocessing**
Create `metadata/preprocessing_steps.json`:
```json
{
  "version": "1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "steps": [
    {"step": "drop_missing", "columns": ["age"], "count": 12},
    {"step": "standardize_numeric", "columns": ["income", "score"], "method": "z-score"},
    {"step": "encode_categorical", "columns": ["region"], "method": "one-hot"}
  ],
  "train_test_split": {"ratio": 0.8, "seed": 42}
}
```

### 3. **Data Statistics & Validation**
Create `metadata/data_statistics.json`:
```json
{
  "dataset": "classification_v1",
  "row_count": 850,
  "feature_count": 12,
  "missing_values": 0,
  "class_distribution": {"0": 0.55, "1": 0.45},
  "numeric_ranges": {"age": [18, 75], "income": [15000, 200000]}
}
```

### 4. **Version Control**
- Include `.gitignore` entry: `processed/**/*.csv` (for large files)
- Commit metadata JSON files to version control
- Add `.gitkeep` in empty subfolders

```gitignore
# Ignore processed CSV files (large)
data/processed/**/*.csv

# Keep metadata tracked
!data/processed/metadata/
!data/processed/**/*.json
```

### 5. **Reproducibility**
Document feature engineering in Jupyter notebooks:
```python
# notebooks/preprocessing.ipynb
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# Save processed data
pd.DataFrame(X_scaled, columns=X_train.columns).to_csv(
    'data/processed/train_features.csv', index=False)
```

## Workflow Integration

1. **Data Ingestion** → `data/raw/` (unmodified raw files)
2. **Data Cleaning** → Remove missing values, handle outliers (in notebook)
3. **Feature Engineering** → Create derived features (in notebook)
4. **Data Export** → Save to `data/processed/` with metadata
5. **Model Training** → Load from `data/processed/` for reproducible results

## Common Pitfalls to Avoid

- ❌ Mixing train/test data before splitting
- ❌ Losing track of preprocessing steps (always document!)
- ❌ Using different preprocessing for train vs. test sets
- ❌ Committing large CSV files (use .gitignore)
- ✅ Version processed data with clear naming (e.g., `v1_2024-01-15.csv`)

## Tools & Libraries

- **Pandas**: `df.to_csv()`, `df.read_csv()`
- **Scikit-learn**: `StandardScaler`, `OneHotEncoder` for feature scaling
- **Joblib**: Save preprocessing pipelines for reuse: `joblib.dump(scaler, 'scaler.pkl')`
