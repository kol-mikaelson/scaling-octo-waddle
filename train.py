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
# CONFIGURATION - Modify these variables for each experiment
# ============================================================================

# Model Configuration
MODEL_TYPE = "RandomForest"  # Options: LinearRegression, Ridge, Lasso, RandomForest
HYPERPARAMETERS = {
    # For LinearRegression: no hyperparameters
    # For Ridge: alpha (default 1.0)
    # For Lasso: alpha (default 1.0)
    # For RandomForest: n_estimators, max_depth
    "n_estimators": 100,
    "max_depth": 15,
}

# Preprocessing Configuration
PREPROCESSING = "none"  # Options: none, standardization, minmax
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature Selection
FEATURE_SELECTION = "correlation"  # Options: all, correlation
CORRELATION_THRESHOLD = 0.1

# ============================================================================
# SETUP
# ============================================================================

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

print("="*80)
print("WINE QUALITY MODEL TRAINING")
print("="*80)
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# LOAD AND EXPLORE DATASET
# ============================================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=';')

print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"Target variable: quality")

# ============================================================================
# FEATURE SELECTION
# ============================================================================

X = df.drop('quality', axis=1)
y = df['quality']

if FEATURE_SELECTION == "correlation":
    correlations = df.corr()['quality'].abs()
    selected_features = correlations[correlations > CORRELATION_THRESHOLD].index.tolist()
    selected_features.remove('quality')
    X = X[selected_features]
    print(f"\nFeature Selection: Correlation-based (threshold={CORRELATION_THRESHOLD})")
    print(f"Selected {len(selected_features)} features: {selected_features}")
else:
    print(f"\nFeature Selection: All features ({X.shape[1]} features)")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\nTrain-Test Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# PREPROCESSING
# ============================================================================

if PREPROCESSING == "standardization":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"\nPreprocessing: StandardScaler applied")
elif PREPROCESSING == "minmax":
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"\nPreprocessing: MinMaxScaler applied")
else:
    print(f"\nPreprocessing: None")

# ============================================================================
# MODEL TRAINING
# ============================================================================

print(f"\nModel Type: {MODEL_TYPE}")
print(f"Hyperparameters: {HYPERPARAMETERS}")

if MODEL_TYPE == "LinearRegression":
    model = LinearRegression()
elif MODEL_TYPE == "Ridge":
    model = Ridge(alpha=HYPERPARAMETERS.get("alpha", 1.0))
elif MODEL_TYPE == "Lasso":
    model = Lasso(alpha=HYPERPARAMETERS.get("alpha", 1.0))
elif MODEL_TYPE == "RandomForest":
    model = RandomForestRegressor(
        n_estimators=HYPERPARAMETERS.get("n_estimators", 100),
        max_depth=HYPERPARAMETERS.get("max_depth", 15),
        random_state=RANDOM_STATE
    )
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# EVALUATION
# ============================================================================

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*80)
print("EVALUATION METRICS")
print("="*80)
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R² Score: {r2:.6f}")
print("="*80)

# ============================================================================
# SAVE MODEL
# ============================================================================

model_path = "output/model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved to: {model_path}")

# ============================================================================
# SAVE RESULTS TO JSON
# ============================================================================

results = {
    "model_type": MODEL_TYPE,
    "hyperparameters": HYPERPARAMETERS,
    "preprocessing": PREPROCESSING,
    "feature_selection": FEATURE_SELECTION,
    "correlation_threshold": CORRELATION_THRESHOLD,
    "test_size": TEST_SIZE,
    "num_features": X.shape[1],
    "num_samples": len(df),
    "metrics": {
        "mse": float(mse),
        "r2_score": float(r2)
    },
    "timestamp": datetime.now().isoformat()
}

results_path = "output/results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Results saved to: {results_path}")

print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")