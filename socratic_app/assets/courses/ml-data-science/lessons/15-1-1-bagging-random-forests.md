# Bagging and Random Forests

## Introduction

Bagging (Bootstrap Aggregating) and Random Forests are ensemble methods that combine multiple models to improve prediction accuracy and reduce overfitting.

## The Ensemble Idea

```python
import numpy as np
import pandas as pd

print("=== ENSEMBLE LEARNING ===")
print("""
WISDOM OF CROWDS:

Individual model: Might have errors
Ensemble: Aggregate predictions → better accuracy

Why ensembles work:
  1. Reduce variance (averaging reduces noise)
  2. Different models make different errors
  3. Errors cancel out when combined

Types of ensembles:
  - BAGGING: Same algorithm, different data samples
  - BOOSTING: Sequential error correction
  - STACKING: Different algorithms, meta-learner
""")

# Demonstrate variance reduction
np.random.seed(42)
true_value = 10
n_models = 100
model_predictions = true_value + np.random.randn(n_models) * 3  # Noisy predictions

print("Variance reduction with ensemble:")
for n in [1, 5, 10, 50, 100]:
    ensemble_pred = np.mean(model_predictions[:n])
    error = abs(ensemble_pred - true_value)
    print(f"  {n:3d} models: prediction = {ensemble_pred:.2f}, error = {error:.2f}")
```

## Bagging (Bootstrap Aggregating)

```python
print("\n=== BAGGING ===")
print("""
PROCEDURE:

1. Create B bootstrap samples from training data
   - Sample WITH replacement
   - Each sample same size as original
   
2. Train one model on each bootstrap sample
   
3. Aggregate predictions:
   - Classification: Majority vote
   - Regression: Average

Bootstrap sample properties:
  - ~63.2% unique samples (on average)
  - ~36.8% are duplicates
  - Different samples → different models
""")

def bootstrap_sample(X, y, random_state=None):
    """Create a bootstrap sample"""
    rng = np.random.RandomState(random_state)
    n = len(X)
    indices = rng.choice(n, size=n, replace=True)
    return X[indices], y[indices], indices

# Example
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

print("\nOriginal data indices: [0, 1, 2, 3, 4]")
for i in range(3):
    _, _, indices = bootstrap_sample(X, y, random_state=i)
    unique_frac = len(np.unique(indices)) / len(indices)
    print(f"Bootstrap {i+1}: {indices} ({unique_frac:.0%} unique)")
```

## Out-of-Bag (OOB) Evaluation

```python
print("\n=== OUT-OF-BAG EVALUATION ===")
print("""
OOB samples = data NOT in a bootstrap sample

For each data point:
  - Some models didn't see it (~1/3 of models)
  - Use those models to predict
  - Compare with true label

OOB score:
  - Free validation without held-out set!
  - Similar to cross-validation
  - Available in sklearn: oob_score=True

OOB Error ≈ Test Error (typically)
""")

print("""
Example: 3 bootstrap samples, 5 data points

             Sample 1   Sample 2   Sample 3   OOB predictions
Point 0      included   excluded   included   Model 2
Point 1      excluded   included   included   Model 1
Point 2      included   included   excluded   Model 3
Point 3      excluded   excluded   included   Models 1, 2
Point 4      included   excluded   excluded   Models 2, 3

OOB score = accuracy on OOB predictions
""")
```

## Random Forests

```python
print("\n=== RANDOM FORESTS ===")
print("""
Random Forest = Bagging + Random Feature Selection

Extra randomization at each split:

1. Bootstrap sample (like bagging)
2. At each node, consider only √p features (classification)
   or p/3 features (regression)
3. Select best split among random subset

Why random feature selection?
  - Decorrelates the trees
  - Strong features don't dominate all trees
  - More diversity → better ensemble

If one feature is very strong:
  - Bagging: All trees use it at root → similar trees
  - RF: Some trees can't use it → more diverse
""")

print("""
Random Forest vs Bagging:

Bagging:
  - Consider all features at each split
  - Trees are correlated (similar structure)
  
Random Forest:
  - Random feature subset at each split
  - Trees are decorrelated
  - Usually better performance
""")
```

## Sklearn Implementation

```python
print("\n=== SKLEARN RANDOM FOREST ===")
print("""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification

# Create data
X, y = make_classification(n_samples=1000, n_features=20,
                          n_informative=10, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=10,           # Max tree depth
    max_features='sqrt',    # Features per split
    min_samples_split=5,    # Min samples to split
    min_samples_leaf=2,     # Min samples in leaf
    bootstrap=True,         # Use bootstrap (default)
    oob_score=True,         # Compute OOB score
    n_jobs=-1,              # Use all cores
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate
print(f"Train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"Test accuracy: {rf.score(X_test, y_test):.3f}")
print(f"OOB score: {rf.oob_score_:.3f}")
""")
```

## Feature Importance

```python
print("\n=== FEATURE IMPORTANCE ===")
print("""
Random Forests provide feature importance:

1. GINI IMPORTANCE (default):
   - Total decrease in impurity from splits on feature
   - Summed over all trees
   - Normalized to sum to 1

2. PERMUTATION IMPORTANCE:
   - Shuffle feature values
   - Measure decrease in accuracy
   - More reliable, but slower
""")

print("""
# Gini importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i in range(10):
    print(f"  {i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")

# Permutation importance (more reliable)
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, 
                               n_repeats=10, random_state=42)

print("Permutation importance:")
for i in result.importances_mean.argsort()[::-1][:5]:
    print(f"  Feature {i}: {result.importances_mean[i]:.4f} "
          f"± {result.importances_std[i]:.4f}")
""")

print("""
Gini importance caveats:
  - Biased toward high cardinality features
  - Biased toward correlated features
  - Use permutation importance for reliability
""")
```

## Hyperparameter Tuning

```python
print("\n=== KEY HYPERPARAMETERS ===")
print("""
1. n_estimators (number of trees):
   - More is usually better
   - Diminishing returns after 100-500
   - More trees = slower but more stable

2. max_depth:
   - Deeper = more complex (overfit risk)
   - None (default) = fully grown trees
   - Start with None, reduce if overfitting

3. max_features:
   - 'sqrt': √n_features (classification)
   - 'log2': log₂(n_features)
   - 0.5: 50% of features
   - Lower = more diversity, less accuracy per tree

4. min_samples_split / min_samples_leaf:
   - Higher values = simpler trees
   - Regularization effect

5. bootstrap:
   - True: Standard RF (with replacement)
   - False: Each tree sees all data (more variance)
""")

param_effects = pd.DataFrame({
    'Parameter': ['n_estimators↑', 'max_depth↑', 'max_features↑', 
                  'min_samples_leaf↑'],
    'Bias': ['Same', '↓', '↓', '↑'],
    'Variance': ['↓', '↑', '↑', '↓'],
    'Training Time': ['↑', '↑', '↑', '↓']
})
print("\nParameter effects:")
print(param_effects.to_string(index=False))
```

## Random Forest Advantages

```python
print("\n=== ADVANTAGES & LIMITATIONS ===")
print("""
ADVANTAGES:
✓ Handles high dimensions well
✓ Robust to overfitting
✓ No feature scaling needed
✓ Handles missing values (some implementations)
✓ Provides feature importance
✓ Parallelizable (fast training)
✓ Good out-of-the-box performance

LIMITATIONS:
✗ Not interpretable (many trees)
✗ Large model size
✗ Slower prediction than single tree
✗ Can struggle with extrapolation
✗ Biased importance for correlated features

WHEN TO USE:
- Structured/tabular data
- When interpretability not critical
- As a baseline before trying complex models
- Feature selection (via importance)
""")
```

## Key Points

- **Bagging**: Bootstrap samples + aggregate predictions
- **Random Forest**: Bagging + random feature subsets
- **OOB score**: Free validation using unseen samples
- **Feature importance**: Gini or permutation-based
- **Decorrelation**: Random features make trees diverse
- **Robust**: Rarely overfits with enough trees

## Reflection Questions

1. Why does bootstrap sampling help reduce variance?
2. How does random feature selection decorrelate the trees?
3. When would you prefer a single decision tree over Random Forest?
