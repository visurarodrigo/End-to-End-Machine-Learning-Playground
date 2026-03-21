# Jupyter Notebooks - Data Exploration & Analysis

This folder contains Jupyter notebooks for exploratory data analysis and model experimentation.

## Notebooks

### `exploration.ipynb`
Comprehensive end-to-end ML workflow notebook:

**Sections:**
1. **Setup** - Imports and data loading
2. **EDA** - Exploratory data analysis with statistics
3. **Feature Analysis** - Distributions, correlations, class balance
4. **Visualization** - Histograms, heatmaps, bar charts
5. **Model Training** - Logistic Regression vs Random Forest
6. **Evaluation** - Confusion matrices, classification reports
7. **Feature Importance** - Random forest feature rankings
8. **Summary** - Side-by-side model comparison

## Running Notebooks

### Prerequisites
```bash
pip install jupyter matplotlib seaborn
```

### Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

Then navigate to `notebooks/exploration.ipynb`

## Best Practices

- **Organize by Topic** - Create separate notebooks for different experiments
- **Clear Documentation** - Use markdown cells to explain analysis steps
- **Clean Outputs** - Clear notebook outputs before committing:
  ```
  Kernel → Restart & Clear Output
  ```
- **Extract Code** - Move reusable code to `app/services` or `app/utils`
- **Version Control** - Avoid committing large notebook files with outputs
- **Self-Contained** - Each notebook should stand alone with data loading

## Creating New Notebooks

Template structure:
```markdown
# [Analysis Title]

## 1. Setup and Imports
## 2. Load Data
## 3. Exploratory Analysis
## 4. Feature Engineering
## 5. Model Training
## 6. Evaluation
## 7. Conclusions
```

## Tips

- Use `%matplotlib inline` for inline plots
- Use descriptive cell comments
- Keep cells focused on single tasks
- Document findings and insights
- Export important visualizations as images
