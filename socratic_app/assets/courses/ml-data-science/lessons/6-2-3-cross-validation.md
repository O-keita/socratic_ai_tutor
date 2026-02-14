# Cross-Validation and Model Selection

## Introduction

Cross-validation provides reliable estimates of model performance by systematically using different portions of data for training and testing. It's essential for model selection and hyperparameter tuning.

## Why Cross-Validation?

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score, KFold,
                                     StratifiedKFold, LeaveOneOut, GridSearchCV)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import make_classification

np.random.seed(42)

print("=== WHY CROSS-VALIDATION? ===")
print("""
Problems with single train/test split:
  1. Estimates vary with different splits
  2. Not using all data for evaluation
  3. May get lucky/unlucky with split

Cross-validation solves this:
  - Systematic evaluation on multiple splits
  - Every point gets to be in test set
  - More reliable performance estimate
  - Better understanding of variance
""")

# Demonstrate variance in single split
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

print("Accuracy with different random splits:")
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"  Split {i}: {acc:.3f}")

print("\nNotice the variance! Cross-validation gives more stable estimates.")
```

## K-Fold Cross-Validation

```python
print("\n=== K-FOLD CROSS-VALIDATION ===")
print("""
Procedure (K=5 example):
  1. Split data into 5 equal folds
  2. For each fold i = 1 to 5:
     - Use fold i as test set
     - Use remaining 4 folds as training set
     - Train model, record test score
  3. Average scores across all folds

Data flow (5-fold):
  Fold 1: [TEST] [Train] [Train] [Train] [Train]
  Fold 2: [Train] [TEST] [Train] [Train] [Train]
  Fold 3: [Train] [Train] [TEST] [Train] [Train]
  Fold 4: [Train] [Train] [Train] [TEST] [Train]
  Fold 5: [Train] [Train] [Train] [Train] [TEST]
""")

# Simple cross-validation
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=5)

print(f"5-Fold CV Results:")
print(f"  Scores: {scores.round(3)}")
print(f"  Mean: {scores.mean():.3f}")
print(f"  Std: {scores.std():.3f}")
print(f"  95% CI: {scores.mean():.3f} ± {1.96*scores.std():.3f}")
```

## Stratified K-Fold

```python
print("\n=== STRATIFIED K-FOLD ===")
print("""
For classification: Preserve class proportions in each fold.

Important when:
  - Imbalanced classes
  - Small dataset
  - Need consistent class distribution
""")

# Create imbalanced data
X_imb, y_imb = make_classification(n_samples=200, n_features=10, 
                                   weights=[0.9, 0.1], random_state=42)

print(f"Class distribution: {np.bincount(y_imb)}")

# Regular K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\nRegular K-Fold class distribution per fold:")
for i, (train_idx, test_idx) in enumerate(kf.split(X_imb)):
    print(f"  Fold {i}: Train {np.bincount(y_imb[train_idx])}, Test {np.bincount(y_imb[test_idx])}")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nStratified K-Fold (balanced):")
for i, (train_idx, test_idx) in enumerate(skf.split(X_imb, y_imb)):
    print(f"  Fold {i}: Train {np.bincount(y_imb[train_idx])}, Test {np.bincount(y_imb[test_idx])}")
```

## Other Cross-Validation Strategies

```python
print("\n=== OTHER CV STRATEGIES ===")
print("""
LEAVE-ONE-OUT (LOO):
  - K = n (each sample is a fold)
  - Maximum use of training data
  - Very computationally expensive
  - High variance in estimates

LEAVE-P-OUT:
  - Leave p samples out each time
  - Even more expensive

REPEATED K-FOLD:
  - Repeat K-fold multiple times
  - Different random splits each time
  - More reliable estimates

TIME SERIES SPLIT:
  - For temporal data
  - Training always before test
  - No data leakage
""")

from sklearn.model_selection import RepeatedKFold, TimeSeriesSplit

# Repeated K-Fold
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
scores_rkf = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=rkf)
print(f"Repeated 5-Fold (3 repeats):")
print(f"  Mean: {scores_rkf.mean():.3f}, Std: {scores_rkf.std():.3f}")

# Time Series Split
print("\nTime Series Split (5 splits):")
tss = TimeSeriesSplit(n_splits=5)
for i, (train_idx, test_idx) in enumerate(tss.split(X)):
    print(f"  Split {i}: Train {len(train_idx)}, Test {len(test_idx)}")
```

## Hyperparameter Tuning with Grid Search

```python
print("\n=== GRID SEARCH CV ===")
print("""
Systematically search hyperparameter combinations:
  1. Define parameter grid
  2. For each combination:
     - Run cross-validation
     - Record mean score
  3. Select best combination
""")

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Grid Search with CV
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, solver='saga'),
    param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)
grid_search.fit(X, y)

print("Grid Search Results:")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.3f}")

# Results DataFrame
results = pd.DataFrame(grid_search.cv_results_)
print("\nTop 5 parameter combinations:")
print(results[['param_C', 'param_penalty', 'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False)
      .head().to_string(index=False))
```

## Random Search

```python
print("\n=== RANDOM SEARCH CV ===")
print("""
Instead of exhaustive grid:
  - Sample random combinations
  - More efficient for large spaces
  - Can explore continuous ranges

Often finds good solutions faster than grid search.
""")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# Define distributions
param_distributions = {
    'C': loguniform(1e-3, 1e3),  # Log-uniform distribution
    'penalty': ['l1', 'l2']
}

random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, solver='saga'),
    param_distributions,
    n_iter=20,  # Number of random samples
    cv=5,
    scoring='accuracy',
    random_state=42
)
random_search.fit(X, y)

print("Random Search Results:")
print(f"  Best parameters: {random_search.best_params_}")
print(f"  Best CV score: {random_search.best_score_:.3f}")
```

## Nested Cross-Validation

```python
print("\n=== NESTED CROSS-VALIDATION ===")
print("""
For unbiased performance estimate when tuning:

PROBLEM: Using CV for both tuning AND evaluation
  - Performance estimate is optimistically biased
  - Model "sees" test data indirectly

SOLUTION: Nested CV
  - Outer loop: Evaluate performance
  - Inner loop: Tune hyperparameters

Structure:
  Outer CV: Split data for final evaluation
    Inner CV: Tune hyperparameters on training fold
""")

from sklearn.model_selection import cross_val_score

# Inner CV for hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Grid search (inner loop)
grid_search_inner = GridSearchCV(
    LogisticRegression(max_iter=1000, solver='saga'),
    {'C': [0.1, 1, 10]},
    cv=inner_cv,
    scoring='accuracy'
)

# Outer CV for performance evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Nested CV scores
nested_scores = cross_val_score(grid_search_inner, X, y, cv=outer_cv)

print("Nested CV Results:")
print(f"  Scores: {nested_scores.round(3)}")
print(f"  Mean: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
print("\nThis gives unbiased estimate of generalization performance.")
```

## Choosing K

```python
print("\n=== CHOOSING K ===")
print("""
COMMON CHOICES:

K = 5 or K = 10:
  - Good balance of bias and variance
  - Widely used, well-understood
  - Reasonable computation time

K = N (Leave-One-Out):
  - Minimum bias (uses almost all data)
  - High variance
  - Expensive for large N

CONSIDERATIONS:
  - Larger K: Lower bias, higher variance, slower
  - Smaller K: Higher bias, lower variance, faster

RECOMMENDATIONS:
  - Start with K = 5 or 10
  - Use stratified for classification
  - Use repeated CV for small datasets
  - Use time series split for temporal data
""")
```

## Key Points

- **Cross-validation**: More reliable than single train/test split
- **K-Fold**: Split into K parts, rotate test set
- **Stratified**: Preserve class proportions (classification)
- **Grid Search**: Exhaustive hyperparameter search
- **Random Search**: Efficient for large parameter spaces
- **Nested CV**: Unbiased performance estimate with tuning
- **K = 5 or 10**: Common choices, balance bias/variance

## Reflection Questions

1. Why might a single train/test split give misleading results?
2. When would you use Leave-One-Out instead of 5-fold CV?
3. Why is nested CV important when tuning hyperparameters?
