# Lab 2: Wine Quality Model Training with GitHub Actions

## Overview
Automated ML workflow using GitHub Actions for training and evaluating wine quality models.

## Dataset
UCI Wine Quality Dataset (White Wine)
- Source: https://archive.ics.uci.edu/dataset/186/wine+quality
- Auto-downloaded during training

## How to Run Experiments

1. Edit `train.py` configuration section at the top:
   - Change `MODEL_TYPE`
   - Modify `HYPERPARAMETERS`
   - Update `PREPROCESSING`
   - Adjust `FEATURE_SELECTION`

2. Commit with meaningful message:
```bash
   git add train.py
   git commit -m "Model-RandomForest, n_estimators-100, max_depth-15, preprocessing-none"
```

3. Push to trigger workflow:
```bash
   git push origin main
```

4. View results in GitHub Actions tab

## Configuration Variables in train.py
```python
MODEL_TYPE = "RandomForest"  # LinearRegression, Ridge, Lasso, RandomForest
HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 15,
}
PREPROCESSING = "none"  # none, standardization, minmax
FEATURE_SELECTION = "correlation"  # all, correlation
```