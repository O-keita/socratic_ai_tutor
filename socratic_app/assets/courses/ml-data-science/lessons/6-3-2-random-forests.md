# Random Forests

## Introduction

Random forests combine multiple decision trees to create a powerful ensemble model that reduces overfitting and improves prediction accuracy.

## Core Concepts

### The Ensemble Approach

Random forests use two key ideas:
1. **Bagging**: Train trees on random subsets of data
2. **Feature randomness**: Consider random features at each split

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # use all CPU cores
)
rf_clf.fit(X_train, y_train)

# Regression
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train, y_train)
```

### How Random Forests Work

1. Create N bootstrap samples (random samples with replacement)
2. Build a decision tree on each sample
3. At each split, consider only a random subset of features
4. Aggregate predictions:
   - Classification: majority vote
   - Regression: average

### Key Hyperparameters

```python
rf = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=15,           # Max depth per tree
    min_samples_split=10,   # Min samples to split
    min_samples_leaf=4,     # Min samples in leaf
    max_features='sqrt',    # Features per split
    bootstrap=True,         # Use bootstrap samples
    oob_score=True,         # Calculate out-of-bag score
    random_state=42
)
```

### Out-of-Bag Error

Each tree is trained on ~63% of data (bootstrap). The remaining 37% can validate:

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X, y)

print(f"OOB Score: {rf.oob_score_:.4f}")
# This estimates generalization error without a separate test set
```

### Feature Importance

```python
# Get importances
importances = rf.feature_importances_

# Visualize
import matplotlib.pyplot as plt

sorted_idx = np.argsort(importances)
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), 
           [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### Permutation Importance

More reliable importance measure:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, 
                                n_repeats=10, 
                                random_state=42)

for i in result.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]}: "
          f"{result.importances_mean[i]:.3f} "
          f"+/- {result.importances_std[i]:.3f}")
```

### When to Use Random Forests

**Good for:**
- Tabular data with mixed feature types
- When interpretability (via feature importance) matters
- Datasets with many features
- Avoiding overfitting

**Less ideal for:**
- Very high-dimensional sparse data
- When inference speed is critical
- When you need smooth predictions

---

## Key Points

- Random forests reduce variance through ensemble averaging
- Bootstrap sampling + feature randomness creates diversity
- OOB score provides free validation
- Feature importance shows which variables matter most
- Generally robust with little tuning required

---

## Reflection Questions

1. **Think**: Why does averaging many trees reduce overfitting? What role does diversity play?

2. **Consider**: How does `max_features` affect the correlation between trees? What happens if max_features equals the total number of features?

3. **Explore**: When would out-of-bag error be misleading? What assumptions does it make about the data?
