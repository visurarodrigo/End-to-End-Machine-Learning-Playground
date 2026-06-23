# Streamlit ML Playground

Interactive machine learning app built with Streamlit and scikit-learn.  
Train, compare, and visualize ML models directly in your browser — no backend server required.

🚀 **Live Demo:** [ml-playground-visura.streamlit.app](https://ml-playground-visura.streamlit.app)

---

## 💡 Overview
Training and comparing multiple machine learning models usually requires writing repetitive boilerplate code. The **Streamlit ML Playground** is a zero-code, interactive web application that allows users to instantly benchmark regression, classification, and clustering algorithms on any tabular dataset. It is designed for rapid prototyping, educational purposes, and quick data exploration.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Multi-page app architecture)
- **Backend/ML:** Python, Scikit-learn, Pandas, NumPy
- **Visualization:** Plotly / Matplotlib / Seaborn

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Pages

### 📁 Upload
- Upload your own CSV or load a built-in sample dataset
- Instant data preview — row count, column types, missing values
- Three sample datasets ready to use: House Prices, Loan Approval, Clustering

### 📉 Regression
- Trains 5 models in one click: Linear, Scaled Linear, Polynomial, Ridge, Lasso
- MSE comparison bar chart with best model highlighted
- Actual vs Predicted scatter plot
- Plain-English interpretation of scaling, polynomial, and regularization results

### 🎯 Classification
- Models: Logistic Regression, Decision Tree (tunable depth), Random Forest
- Metrics: Accuracy, Precision, Recall, F1 Score
- Confusion matrix heatmaps for each model
- Train vs Test accuracy comparison — overfitting detection built in

### 🔍 Unsupervised
- **KMeans:** Adjustable k, cluster distribution chart, 2D scatter by cluster
- **PCA:** Configurable components, explained variance chart, PC1 vs PC2 scatter
- All numeric features used automatically — no configuration needed

---

## Sample Datasets

| Dataset | Rows | Features | Best For |
|---|---|---|---|
| 🏠 House Prices | 50 | 6 numeric | Regression |
| 💳 Loan Approval | 50 | 6 numeric, binary target | Classification |
| 🔵 Clustering | 50 | 4 numeric | KMeans / PCA |

---

## Folder Structure

```
streamlit_app/
├── app.py                  # Home page
├── requirements.txt        # Dependencies
├── pages/
│   ├── 1_Upload.py         # Upload & preview
│   ├── 2_Regression.py     # Regression models
│   ├── 3_Classification.py # Classification models
│   └── 4_Unsupervised.py   # KMeans & PCA
└── utils/
    ├── api_client.py       # ML service layer (direct scikit-learn calls)
    └── sample_data.py      # Sample dataset loader
```

---

## Screenshots

### Home
![Home Page](streamlit_app/Screen%20Shots/home%20page.png)

### Upload
![Upload Page](streamlit_app/Screen%20Shots/upload%20csv%20page.png)

### Regression
![Regression Page](streamlit_app/Screen%20Shots/Regression%20page.png)

### Classification
![Classification Page](streamlit_app/Screen%20Shots/Classification%20page.png)

### Unsupervised
![Unsupervised Page](streamlit_app/Screen%20Shots/Unsupervised%20page.png)

---

## Troubleshooting

**CSV upload failed** — make sure the file is `.csv` format with numeric columns. Try a sample dataset first to confirm the app is working.

**Classification target error** — choose a column with a small number of unique values (binary or categorical). Continuous numeric columns won't work well as classification targets.

**Clustering needs numeric data** — if your dataset has only text columns, the unsupervised page will flag it.

---

👤 Connect with Me
Built by **Visura Rodrigo**  


