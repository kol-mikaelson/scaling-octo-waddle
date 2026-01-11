import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime

# ============================================================================
# LOAD AND EXPLORE DATASET
# ============================================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=';')

print("="*80)
print("WINE QUALITY MODEL TRAINING - ALL EXPERIMENTS")
print("="*80)
print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"Target variable: quality\n")

# Prepare data once
X = df.drop('quality', axis=1)
y = df['quality']

# Create output directory
os.makedirs("output", exist_ok=True)

# ============================================================================
# DEFINE ALL EXPERIMENTS
# ============================================================================

experiments = [
    {
        "id": "EXP-01",
        "name": "Linear Regression (Default)",
        "model_type": "LinearRegression",
        "hyperparameters": {},
        "preprocessing": "none",
        "feature_selection": "all",
        "correlation_threshold": 0.1,
        "test_size": 0.2,
    },
    {
        "id": "EXP-02",
        "name": "Ridge Regression (Standardization + Correlation)",
        "model_type": "Ridge",
        "hyperparameters": {"alpha": 1.0},
        "preprocessing": "standardization",
        "feature_selection": "correlation",
        "correlation_threshold": 0.1,
        "test_size": 0.2,
    },
    {
        "id": "EXP-03",
        "name": "Random Forest (50 trees, depth=10)",
        "model_type": "RandomForest",
        "hyperparameters": {"n_estimators": 50, "max_depth": 10},
        "preprocessing": "none",
        "feature_selection": "all",
        "correlation_threshold": 0.1,
        "test_size": 0.2,
    },
    {
        "id": "EXP-04",
        "name": "Random Forest (100 trees, depth=15) + Features",
        "model_type": "RandomForest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 15},
        "preprocessing": "none",
        "feature_selection": "correlation",
        "correlation_threshold": 0.1,
        "test_size": 0.2,
    },
]

# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

results_list = []

for exp in experiments:
    print("="*80)
    print(f"{exp['id']}: {exp['name']}")
    print("="*80)
    
    # Feature Selection
    if exp['feature_selection'] == "correlation":
        correlations = df.corr()['quality'].abs()
        selected_features = correlations[correlations > exp['correlation_threshold']].index.tolist()
        selected_features.remove('quality')
        X_selected = X[selected_features]
        print(f"Feature Selection: Correlation-based (threshold={exp['correlation_threshold']})")
        print(f"Selected {len(selected_features)} features: {selected_features}")
    else:
        X_selected = X.copy()
        print(f"Feature Selection: All features ({X_selected.shape[1]} features)")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=exp['test_size'], random_state=42
    )
    print(f"Train-Test Split: {int((1-exp['test_size'])*100)}/{int(exp['test_size']*100)}")
    
    # Preprocessing
    if exp['preprocessing'] == "standardization":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print(f"Preprocessing: StandardScaler applied")
    elif exp['preprocessing'] == "minmax":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print(f"Preprocessing: MinMaxScaler applied")
    else:
        print(f"Preprocessing: None")
    
    # Model Training
    print(f"Hyperparameters: {exp['hyperparameters']}")
    
    if exp['model_type'] == "LinearRegression":
        model = LinearRegression()
    elif exp['model_type'] == "Ridge":
        model = Ridge(alpha=exp['hyperparameters'].get("alpha", 1.0))
    elif exp['model_type'] == "Lasso":
        model = Lasso(alpha=exp['hyperparameters'].get("alpha", 1.0))
    elif exp['model_type'] == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=exp['hyperparameters'].get("n_estimators", 100),
            max_depth=exp['hyperparameters'].get("max_depth", 15),
            random_state=42
        )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2:.6f}\n")
    
    # Store results
    result = {
        "experiment_id": exp['id'],
        "model_type": exp['model_type'],
        "hyperparameters": exp['hyperparameters'],
        "preprocessing": exp['preprocessing'],
        "feature_selection": exp['feature_selection'],
        "correlation_threshold": exp['correlation_threshold'],
        "test_size": exp['test_size'],
        "num_features": X_selected.shape[1],
        "num_samples": len(df),
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "metrics": {
            "mse": float(mse),
            "r2_score": float(r2)
        },
        "timestamp": datetime.now().isoformat()
    }
    results_list.append(result)

# ============================================================================
# SAVE ALL RESULTS TO JSON
# ============================================================================

results_path = "output/all_results.json"
with open(results_path, 'w') as f:
    json.dump(results_list, f, indent=4)
print(f"All results saved to: {results_path}")

# ============================================================================
# CREATE SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE - ALL EXPERIMENTS")
print("="*80)
print(f"{'Exp ID':<10} {'Model':<20} {'MSE':<12} {'R² Score':<12}")
print("-"*80)
for result in results_list:
    print(f"{result['experiment_id']:<10} {result['model_type']:<20} {result['metrics']['mse']:<12.6f} {result['metrics']['r2_score']:<12.6f}")

# Find best model
best_exp = max(results_list, key=lambda x: x['metrics']['r2_score'])
print("-"*80)
print(f"Best Model: {best_exp['experiment_id']} - {best_exp['model_type']}")
print(f"R² Score: {best_exp['metrics']['r2_score']:.6f}")
print(f"MSE: {best_exp['metrics']['mse']:.6f}")
print("="*80)

print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")