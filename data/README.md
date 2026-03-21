# Data - Datasets & CSV Files

Stores all datasets organized by processing stage.

## Subfolders

### `raw/`
Original, unprocessed CSV files:
- `sample_classification.csv` - Binary classification (~100 rows)
- `sample_regression.csv` - Regression with price target (~100 rows)
- `sample_unsupervised.csv` - Clustering/PCA (~150 rows)

Upload these directly via Swagger UI at `http://127.0.0.1:8000/docs`

### `processed/`
Cleaned and transformed datasets generated during workflows:
- Feature-engineered versions
- Train/test splits
- Intermediate results

## Best Practices

- Keep sample files small (<1MB)
- Add large files to `.gitignore`
- Document transformations with metadata files
- Use consistent CSV format (comma-delimited, UTF-8)
- Include headers and handle missing values consistently
