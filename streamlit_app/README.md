# Streamlit ML Playground

Complete interactive UI for the End-to-End ML Playground backend.

## Setup

### 1. Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Ensure FastAPI Backend is Running

In another terminal, start the FastAPI server:

```bash
cd ..
uvicorn app.main:app --reload
```

The API should be running at `http://127.0.0.1:8000`

### 3. Run Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Features

### 📁 Upload Page
- **CSV Upload** - Drag & drop your dataset
- **Sample Datasets** - One-click test with pre-built datasets
  - 🏠 House Price Prediction (Regression)
  - 💳 Loan Approval (Classification)
  - 🎯 Customer Clustering (Unsupervised)
- **Data Preview** - Schema, missing values, statistics

### 📉 Regression Page
- Train 5 regression models in one click:
  - Linear Regression (baseline)
  - Scaled Linear Regression
  - Polynomial Regression (degree 2)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
- **Metrics:** MSE, RMSE comparisons
- **Visualizations:** Performance bar chart, Actual vs Predicted scatter plot
- **Insights:** Best model highlight, performance messages

### 🎯 Classification Page
- Train up to 4 classification models:
  - Logistic Regression
  - Decision Tree (with tunable max_depth)
  - Random Forest
  - Neural Network (TensorFlow/Keras)
- **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Visualizations:** Model comparison, confusion matrices
- **Insights:** Overfitting detection (accuracy gap)

### 🔍 Unsupervised Page
- **KMeans Clustering** - Adjustable k, cluster distribution
- **PCA Reduction** - Explained variance visualization, cumulative variance
- **Metrics:** Cluster sizes, variance percentages

---

## Architecture

```
streamlit_app/
├── app.py                      # Main Streamlit app (home page)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── pages/
│   ├── 1_Upload.py            # Dataset upload & preview
│   ├── 2_Regression.py        # Regression models
│   ├── 3_Classification.py    # Classification models
│   └── 4_Unsupervised.py      # KMeans & PCA
└── utils/
    ├── __init__.py
    ├── api_client.py          # FastAPI communication
    └── sample_data.py         # Load sample datasets
```

---

## API Integration

All pages communicate with the FastAPI backend via `api_client.py`:

- **Upload:** `/upload` endpoint (regression workflow)
- **Classification:** `/train-classification-*` endpoints
- **Clustering:** `/train-clustering-kmeans`, `/train-pca` endpoints

The app expects the FastAPI server to be running on `http://127.0.0.1:8000`

---

## Using Sample Datasets

Click any sample button on the Upload page to instantly load:

1. **House Prices** (50 rows, 6 numeric columns, target: `price`)
   - Best for testing regression workflows
   
2. **Loan Approval** (50 rows, 6 numeric columns, target: `target` 0/1)
   - Best for testing classification workflows
   
3. **Clustering** (50 rows, 4 numeric features, no target)
   - Best for testing unsupervised learning

---

## Troubleshooting

### "API not running" Error
- Make sure FastAPI is running: `uvicorn app.main:app --reload`
- Check it's at `http://127.0.0.1:8000/docs`

### "CSV Upload Failed"
- Ensure file is `.csv` format
- Check for odd characters in column names
- Try a sample dataset first

### Models training slowly
- This is normal for larger datasets
- Neural Network training takes longer (~10-30 seconds)
- Check your CPU usage

---

## Next Steps

1. ✅ Upload your own CSV
2. ✅ Try regression models
3. ✅ Try classification with different target columns
4. ✅ Explore clustering on numeric data
5. 🔄 Compare results, iterate on parameters

Enjoy exploring ML workflows! 🚀
