# Boosting: AdaBoost and Gradient Boosting

## Introduction

Boosting is an ensemble technique that builds models sequentially, with each new model focusing on the errors of previous models. It's one of the most powerful ML techniques for structured data.

## Boosting vs Bagging

```python
import numpy as np
import pandas as pd

print("=== BOOSTING VS BAGGING ===")
print("""
BAGGING (Parallel):
  - Train models independently
  - Same algorithm, different samples
  - Average predictions
  - Reduces VARIANCE
  
  Model 1 ─┐
  Model 2 ─┼→ Average → Prediction
  Model 3 ─┘
  (parallel, independent)

BOOSTING (Sequential):
  - Train models in sequence
  - Each model corrects previous errors
  - Weighted combination
  - Reduces BIAS
  
  Model 1 → errors → Model 2 → errors → Model 3 → ...
  (sequential, dependent)
""")

print("""
Key differences:
┌─────────────┬──────────────┬──────────────┐
│             │   Bagging    │   Boosting   │
├─────────────┼──────────────┼──────────────┤
│ Training    │   Parallel   │  Sequential  │
│ Reduces     │   Variance   │    Bias      │
│ Trees       │   Full depth │   Shallow    │
│ Overfit     │   Resistant  │  Can overfit │
│ Speed       │   Faster     │   Slower     │
└─────────────┴──────────────┴──────────────┘
""")
```

## AdaBoost

```python
print("\n=== ADABOOST ===")
print("""
Adaptive Boosting - reweight samples based on errors

ALGORITHM:
1. Initialize sample weights: w_i = 1/n for all i
2. For m = 1 to M (number of models):
   a. Train weak learner G_m on weighted data
   b. Compute weighted error: ε_m = Σ w_i × I(y_i ≠ G_m(x_i))
   c. Compute model weight: α_m = 0.5 × log((1-ε_m)/ε_m)
   d. Update sample weights:
      - Increase weight of misclassified samples
      - Decrease weight of correctly classified
3. Final prediction: sign(Σ α_m × G_m(x))

Weak learner: Slightly better than random
  - Often: Decision stump (depth-1 tree)
""")

def adaboost_step(y_true, y_pred, weights):
    """One step of AdaBoost weight update"""
    # Error rate
    incorrect = (y_true != y_pred)
    error = np.sum(weights * incorrect) / np.sum(weights)
    
    # Model weight (higher weight for better models)
    alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
    
    # Update sample weights
    # Increase for misclassified, decrease for correct
    new_weights = weights * np.exp(alpha * (2 * incorrect - 1))
    new_weights = new_weights / np.sum(new_weights)  # Normalize
    
    return alpha, new_weights, error

# Example
y_true = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
y_pred = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1])  # Some errors
weights = np.ones(10) / 10

alpha, new_weights, error = adaboost_step(y_true, y_pred, weights)

print("AdaBoost example:")
print(f"Initial weights: all = {weights[0]:.3f}")
print(f"Error rate: {error:.2f}")
print(f"Model weight (α): {alpha:.3f}")
print(f"\nUpdated weights:")
for i in range(len(y_true)):
    status = "✓" if y_true[i] == y_pred[i] else "✗"
    print(f"  Sample {i}: {status} weight: {weights[i]:.3f} → {new_weights[i]:.3f}")
```

## Gradient Boosting

```python
print("\n=== GRADIENT BOOSTING ===")
print("""
Generalization of boosting using gradients

KEY IDEA: Fit new model to the RESIDUALS (errors)

Algorithm for regression:
1. Initialize F_0(x) = mean(y)
2. For m = 1 to M:
   a. Compute residuals: r_m = y - F_{m-1}(x)
   b. Fit tree h_m to residuals
   c. Update: F_m(x) = F_{m-1}(x) + η × h_m(x)
3. Final: F_M(x)

η = learning rate (shrinkage)
  - Smaller η = more trees needed, but better generalization
  - Typical: 0.01 to 0.1

The "gradient" connection:
  - Residual = negative gradient of MSE loss
  - We're doing gradient descent in function space!
""")

def gradient_boosting_demo(X, y, n_estimators=3, learning_rate=0.5):
    """Simplified gradient boosting demonstration"""
    # Initialize with mean
    predictions = np.full_like(y, np.mean(y), dtype=float)
    print(f"Initial prediction: {np.mean(y):.2f} (mean)")
    
    for m in range(n_estimators):
        # Compute residuals
        residuals = y - predictions
        
        # "Fit" simple model to residuals (just mean for demo)
        update = np.mean(residuals)
        
        # Update predictions
        predictions = predictions + learning_rate * update
        
        mse = np.mean((y - predictions) ** 2)
        print(f"Round {m+1}: residual_mean={np.mean(residuals):.3f}, "
              f"update={update:.3f}, MSE={mse:.3f}")
    
    return predictions

y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
gradient_boosting_demo(None, y)
```

## XGBoost

```python
print("\n=== XGBOOST ===")
print("""
eXtreme Gradient Boosting - optimized and regularized

KEY INNOVATIONS:
1. Second-order gradients (Newton method)
2. Built-in regularization
3. Column subsampling (like RF)
4. Efficient handling of sparse data
5. Parallelization within tree building

Objective:
  L = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
      ├──────┴──────┤   ├──┴──┤
       training loss   regularization

Regularization Ω:
  Ω(f) = γT + 0.5λ||w||²
  T = number of leaves
  w = leaf weights
  γ, λ = regularization parameters
""")

print("""
from xgboost import XGBClassifier, XGBRegressor

model = XGBClassifier(
    n_estimators=100,      # Number of boosting rounds
    learning_rate=0.1,     # Step size shrinkage
    max_depth=6,           # Max tree depth
    min_child_weight=1,    # Min sum of weights in child
    subsample=0.8,         # Row sampling ratio
    colsample_bytree=0.8,  # Column sampling ratio
    gamma=0,               # Min loss reduction for split
    reg_alpha=0,           # L1 regularization
    reg_lambda=1,          # L2 regularization
    objective='binary:logistic',
    eval_metric='logloss'
)

# Training with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=10,
    verbose=False
)
""")
```

## LightGBM

```python
print("\n=== LIGHTGBM ===")
print("""
Light Gradient Boosting Machine - faster training

KEY INNOVATIONS:

1. LEAF-WISE GROWTH (vs level-wise):
   XGBoost: Grow all leaves at same depth
   LightGBM: Grow leaf with max gain
   → Deeper, more accurate trees
   → Risk of overfitting (use max_depth)

2. HISTOGRAM-BASED SPLITTING:
   - Bin continuous features
   - Much faster split finding
   - Slight accuracy trade-off

3. GOSS (Gradient-based One-Side Sampling):
   - Keep samples with large gradients
   - Randomly sample small gradients
   - Focus on hard examples

4. EFB (Exclusive Feature Bundling):
   - Bundle sparse features
   - Reduces dimensionality
""")

print("""
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,          # No limit (use num_leaves)
    num_leaves=31,         # Max leaves per tree
    min_child_samples=20,  # Min samples in leaf
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=0,
    boosting_type='gbdt',  # or 'dart', 'goss'
)

# LightGBM specific: categorical features
model.fit(
    X_train, y_train,
    categorical_feature=[0, 3, 5],  # Column indices
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=10
)
""")
```

## CatBoost

```python
print("\n=== CATBOOST ===")
print("""
Categorical Boosting - handles categories natively

KEY INNOVATIONS:

1. ORDERED BOOSTING:
   - Avoids target leakage
   - Uses random permutations
   - Better generalization

2. NATIVE CATEGORICAL HANDLING:
   - No one-hot encoding needed
   - Target statistics with smoothing
   - Handles high cardinality

3. SYMMETRIC TREES:
   - Same split at each level
   - Fast inference
   - Better GPU utilization
""")

print("""
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    cat_features=[0, 3, 5],  # Categorical column indices
    verbose=False
)

# No preprocessing needed for categoricals!
model.fit(X_train, y_train,
          eval_set=(X_valid, y_valid),
          early_stopping_rounds=10)
""")

print("""
Library comparison:

┌──────────────┬───────────┬──────────┬──────────┐
│ Feature      │  XGBoost  │ LightGBM │ CatBoost │
├──────────────┼───────────┼──────────┼──────────┤
│ Speed        │  Medium   │   Fast   │  Medium  │
│ Accuracy     │   High    │   High   │   High   │
│ Categoricals │    No     │    Yes   │   Best   │
│ Missing      │   Yes     │   Yes    │   Yes    │
│ GPU          │   Yes     │   Yes    │   Yes    │
│ Tree Growth  │ Level     │  Leaf    │ Symmetric│
└──────────────┴───────────┴──────────┴──────────┘
""")
```

## Key Points

- **Boosting**: Sequential models correcting errors
- **AdaBoost**: Reweight samples by errors
- **Gradient Boosting**: Fit residuals (gradient descent in function space)
- **XGBoost**: Regularized, second-order optimization
- **LightGBM**: Leaf-wise growth, histogram binning, fastest
- **CatBoost**: Best native categorical handling

## Reflection Questions

1. Why does boosting reduce bias while bagging reduces variance?
2. How does the learning rate affect the bias-variance trade-off?
3. When would you choose LightGBM over XGBoost?
