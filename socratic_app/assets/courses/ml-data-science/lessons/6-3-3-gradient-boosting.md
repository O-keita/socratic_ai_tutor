# Gradient Boosting Machines

## Introduction

Gradient Boosting builds an ensemble of weak learners (usually decision trees) sequentially, where each new tree corrects the errors of the combined ensemble so far. It's one of the most powerful algorithms for structured data.

## The Gradient Boosting Algorithm

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error

np.random.seed(42)

print("=== GRADIENT BOOSTING ===")
print("""
Key Idea: Build trees sequentially to correct errors

Algorithm:
  1. Initialize with a simple prediction (mean for regression)
  2. For each iteration m = 1 to M:
     a. Compute residuals (errors) from current ensemble
     b. Fit a tree to the residuals
     c. Add tree to ensemble (with learning rate)
  3. Final prediction = sum of all trees

Difference from Random Forest:
  - Random Forest: Trees built independently (parallel)
  - Gradient Boosting: Trees built sequentially (correcting errors)

Mathematical form:
  F_m(x) = F_{m-1}(x) + η * h_m(x)
  
  Where η is learning rate, h_m is the new tree
""")
```

## Gradient Boosting Classification

```python
print("\n=== GRADIENT BOOSTING CLASSIFICATION ===")

# Generate data
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                          n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                    max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)

train_acc = gb_clf.score(X_train, y_train)
test_acc = gb_clf.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Compare with Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
print(f"\nRandom Forest Accuracy: {rf.score(X_test, y_test):.3f}")
print(f"Gradient Boosting Accuracy: {test_acc:.3f}")
```

## Key Hyperparameters

```python
print("\n=== KEY HYPERPARAMETERS ===")
print("""
n_estimators:
  - Number of boosting stages (trees)
  - More trees = better fit, but risk overfitting
  - Usually 100-1000

learning_rate (η):
  - Shrinkage parameter
  - Scales contribution of each tree
  - Smaller = more robust, needs more trees
  - Typical: 0.01 - 0.3

max_depth:
  - Depth of individual trees
  - GB usually uses shallow trees (3-5)
  - Deeper = more complex interactions

subsample:
  - Fraction of samples for each tree
  - < 1.0 adds stochasticity (Stochastic GB)
  - Helps prevent overfitting

min_samples_split / min_samples_leaf:
  - Tree regularization
  - Prevent overfitting
""")

# Effect of learning rate
print("\nEffect of learning_rate:")
for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, 
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    test_acc = gb.score(X_test, y_test)
    print(f"  learning_rate={lr}: {test_acc:.3f}")
```

## Learning Rate and Number of Trees

```python
print("\n=== LEARNING RATE vs N_ESTIMATORS ===")
print("""
Trade-off:
  - Low learning rate → Need more trees → Better generalization
  - High learning rate → Fewer trees needed → Faster but may overfit

Rule of thumb:
  - Use low learning rate (0.01-0.1)
  - Use early stopping to find optimal n_estimators
  - More trees is usually better (with low learning rate)
""")

# Compare combinations
configs = [
    (100, 0.1),
    (500, 0.02),
    (1000, 0.01),
]

print("Comparing configurations:")
for n_est, lr in configs:
    gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                    max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    test_acc = gb.score(X_test, y_test)
    print(f"  n_estimators={n_est:4d}, lr={lr:.2f}: {test_acc:.3f}")
```

## Gradient Boosting Regression

```python
print("\n=== GRADIENT BOOSTING REGRESSION ===")

# Generate regression data
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=50, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
gb_reg.fit(X_train_r, y_train_r)

train_r2 = gb_reg.score(X_train_r, y_train_r)
test_r2 = gb_reg.score(X_test_r, y_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, gb_reg.predict(X_test_r)))

print(f"Training R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
```

## Staged Predictions and Training Progress

```python
print("\n=== MONITORING TRAINING ===")
print("""
Gradient Boosting supports staged predictions:
  - See how error decreases with more trees
  - Useful for finding optimal n_estimators
  - Can detect overfitting
""")

# Track error over iterations
train_errors = []
test_errors = []

gb_monitor = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                        max_depth=3, random_state=42)
gb_monitor.fit(X_train, y_train)

# Staged predictions
for i, (y_train_pred, y_test_pred) in enumerate(zip(
    gb_monitor.staged_predict(X_train),
    gb_monitor.staged_predict(X_test)
)):
    train_err = 1 - accuracy_score(y_train, y_train_pred)
    test_err = 1 - accuracy_score(y_test, y_test_pred)
    train_errors.append(train_err)
    test_errors.append(test_err)

print(f"Error at different stages:")
for n in [10, 50, 100, 150, 200]:
    print(f"  n={n:3d}: Train error={train_errors[n-1]:.3f}, Test error={test_errors[n-1]:.3f}")

# Best number of trees
best_n = np.argmin(test_errors) + 1
print(f"\nOptimal n_estimators: {best_n} (test error: {min(test_errors):.3f})")
```

## Feature Importance

```python
print("\n=== FEATURE IMPORTANCE ===")

# Get feature importances
importances = gb_clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("Top 10 Feature Importances:")
for i in range(10):
    idx = sorted_idx[i]
    print(f"  Feature {idx}: {importances[idx]:.4f}")

print("""
Note: Importance is based on reduction in loss function
  - Higher importance = feature contributes more to predictions
  - Different from permutation importance
""")
```

## Comparison with XGBoost and LightGBM

```python
print("\n=== ADVANCED IMPLEMENTATIONS ===")
print("""
Modern Gradient Boosting Libraries:

XGBoost (Extreme Gradient Boosting):
  - Regularization (L1, L2 on leaf weights)
  - Handling missing values
  - Parallel tree construction
  - Built-in cross-validation
  
LightGBM (Microsoft):
  - Leaf-wise tree growth (faster)
  - Histogram-based binning
  - Categorical feature support
  - Even faster than XGBoost

CatBoost (Yandex):
  - Native categorical handling
  - Ordered boosting (less overfitting)
  - GPU support

Recommendation:
  - Start with sklearn GradientBoosting for learning
  - Use XGBoost or LightGBM for production
  - LightGBM is usually fastest
""")

# Example with XGBoost (if installed)
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=42, verbosity=0)
    xgb_clf.fit(X_train, y_train)
    xgb_acc = xgb_clf.score(X_test, y_test)
    print(f"\nXGBoost Accuracy: {xgb_acc:.3f}")
except ImportError:
    print("\n(XGBoost not installed)")
```

## Advantages and Disadvantages

```python
print("\n=== PROS AND CONS ===")
print("""
ADVANTAGES:
  ✓ Often highest accuracy on structured data
  ✓ Handles mixed feature types
  ✓ Built-in feature importance
  ✓ Robust to outliers (with appropriate loss)
  ✓ No feature scaling needed
  ✓ Can capture complex non-linear relationships

DISADVANTAGES:
  ✗ Slower training (sequential)
  ✗ Memory intensive
  ✗ Prone to overfitting (need careful tuning)
  ✗ Many hyperparameters
  ✗ Not easily interpretable
  ✗ Sensitive to noisy data

WHEN TO USE:
  - Tabular/structured data
  - When accuracy is top priority
  - Competitions (Kaggle favorite)
  
WHEN NOT TO USE:
  - Need real-time training updates
  - Very high-dimensional sparse data
  - Need interpretability
""")
```

## Key Points

- **Sequential ensemble**: Each tree corrects errors of previous ensemble
- **Learning rate**: Shrinks contribution of each tree
- **Shallow trees**: Usually depth 3-5, not full trees
- **Low learning rate**: Use more trees for better generalization
- **Early stopping**: Monitor validation error to prevent overfitting
- **XGBoost/LightGBM**: Faster, more features than sklearn
- **Feature importance**: Built-in, based on split improvements

## Reflection Questions

1. Why does gradient boosting typically use shallow trees while random forests use deeper trees?
2. How does the learning rate affect the bias-variance tradeoff?
3. When would you prefer gradient boosting over a random forest?
