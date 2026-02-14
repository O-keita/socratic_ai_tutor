# Feature Selection Methods

## Introduction

Feature selection identifies the most relevant features for modeling, reducing dimensionality, improving interpretability, and often enhancing performance by removing noise.

## Why Feature Selection?

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

print("=== WHY FEATURE SELECTION? ===")
print("""
Benefits of Feature Selection:

1. IMPROVED PERFORMANCE
   - Removes noisy features
   - Reduces overfitting
   - Can improve accuracy

2. FASTER TRAINING
   - Fewer features = faster computation
   - Important for large datasets

3. BETTER INTERPRETABILITY
   - Easier to understand important factors
   - Simpler models to explain

4. REDUCED STORAGE
   - Less data to store and process

5. AVOID CURSE OF DIMENSIONALITY
   - More features need exponentially more data

Types of Methods:
  - Filter: Statistical measures, independent of model
  - Wrapper: Use model performance to select
  - Embedded: Selection built into model training
""")
```

## Filter Methods

```python
print("\n=== FILTER METHODS ===")
print("""
Select features based on statistical measures.
Fast, model-agnostic, but ignore feature interactions.
""")

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=5, n_redundant=5,
                          random_state=42)
feature_names = [f'feature_{i}' for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Informative features: 5, Redundant: 5, Noise: 10")
```

## Variance Threshold

```python
print("\n=== VARIANCE THRESHOLD ===")
print("""
Remove features with low variance.
Low variance = little information.
""")

from sklearn.feature_selection import VarianceThreshold

# Add a constant feature
X_with_constant = np.hstack([X, np.zeros((X.shape[0], 1))])
print(f"Shape before: {X_with_constant.shape}")

# Remove zero variance features
selector = VarianceThreshold(threshold=0)
X_selected = selector.fit_transform(X_with_constant)
print(f"Shape after (variance > 0): {X_selected.shape}")

# Remove low variance features
selector_low = VarianceThreshold(threshold=0.5)
X_selected_low = selector_low.fit_transform(X)
print(f"Shape after (variance > 0.5): {X_selected_low.shape}")

# Show variances
print("\nFeature variances:")
variances = np.var(X, axis=0)
print(pd.Series(variances, index=feature_names).sort_values(ascending=False).head(10).round(3))
```

## Correlation-Based Selection

```python
print("\n=== CORRELATION WITH TARGET ===")
print("""
Select features most correlated with target.
For classification: use point-biserial correlation.
""")

from scipy.stats import pointbiserialr

# Calculate correlations with target
correlations = {}
for i, col in enumerate(feature_names):
    corr, pval = pointbiserialr(y, X[:, i])
    correlations[col] = {'correlation': abs(corr), 'p_value': pval}

corr_df = pd.DataFrame(correlations).T.sort_values('correlation', ascending=False)
print("Top 10 features by correlation with target:")
print(corr_df.head(10).round(4))

print("\n=== REMOVE HIGHLY CORRELATED FEATURES ===")
print("Features correlated with each other are redundant.")

# Correlation matrix between features
corr_matrix = np.corrcoef(X.T)

# Find highly correlated pairs
threshold = 0.8
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(corr_matrix[i, j]) > threshold:
            high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

print(f"Highly correlated pairs (|r| > {threshold}):")
for f1, f2, corr in high_corr_pairs[:5]:
    print(f"  {f1} <-> {f2}: {corr:.3f}")
```

## Chi-Square Test (Categorical)

```python
print("\n=== CHI-SQUARE TEST ===")
print("""
For categorical features with categorical target.
Tests if feature and target are independent.
Higher chi-square = more dependent = more useful.
""")

from sklearn.feature_selection import chi2, SelectKBest

# Create categorical-like features (non-negative)
X_positive = X - X.min(axis=0)  # Make non-negative for chi2

# Select top 10 features
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X_positive, y)

print(f"Selected {X_selected.shape[1]} features")
print("\nChi-square scores:")
scores = pd.Series(selector.scores_, index=feature_names)
print(scores.sort_values(ascending=False).round(2))

# Show which features were selected
selected_mask = selector.get_support()
selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
print(f"\nSelected features: {selected_features}")
```

## ANOVA F-Test

```python
print("\n=== ANOVA F-TEST ===")
print("""
For continuous features with categorical target.
Tests if feature means differ across classes.
""")

from sklearn.feature_selection import f_classif

# F-test
f_scores, p_values = f_classif(X, y)

print("ANOVA F-scores:")
f_df = pd.DataFrame({
    'feature': feature_names,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)
print(f_df.head(10).round(4))

# Select significant features (p < 0.05)
significant = f_df[f_df['p_value'] < 0.05]
print(f"\nSignificant features (p < 0.05): {len(significant)}")
```

## Mutual Information

```python
print("\n=== MUTUAL INFORMATION ===")
print("""
Measures dependency between feature and target.
Captures non-linear relationships (unlike correlation).
MI = 0 means independent, higher = more dependent.
""")

from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)

print("Mutual Information scores:")
mi_df = pd.DataFrame({
    'feature': feature_names,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)
print(mi_df.head(10).round(4))
```

## Wrapper Methods

```python
print("\n=== WRAPPER METHODS ===")
print("""
Use model performance to select features.
More accurate but computationally expensive.

Types:
  - Forward selection: Start empty, add features
  - Backward elimination: Start full, remove features
  - Recursive Feature Elimination (RFE)
""")

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Recursive Feature Elimination
model = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
rfe.fit(X, y)

print("RFE Selected Features:")
rfe_df = pd.DataFrame({
    'feature': feature_names,
    'selected': rfe.support_,
    'ranking': rfe.ranking_
}).sort_values('ranking')
print(rfe_df.head(15))

selected_rfe = [f for f, s in zip(feature_names, rfe.support_) if s]
print(f"\nSelected features: {selected_rfe}")
```

## Embedded Methods

```python
print("\n=== EMBEDDED METHODS ===")
print("""
Feature selection built into model training.
Examples:
  - L1 regularization (Lasso): Drives coefficients to zero
  - Tree feature importances: Based on split decisions
""")

# L1 Regularization (Lasso)
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

print("=== L1 REGULARIZATION (LASSO) ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_scaled, y)

print("Lasso coefficients:")
lasso_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': np.abs(lasso.coef_)
}).sort_values('coefficient', ascending=False)
print(lasso_df.head(10).round(4))
print(f"\nFeatures with non-zero coefficients: {np.sum(lasso.coef_ != 0)}")

# Tree Feature Importance
from sklearn.ensemble import RandomForestClassifier

print("\n=== RANDOM FOREST IMPORTANCE ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

print("Random Forest feature importances:")
rf_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_df.head(10).round(4))
```

## Selection Strategy Guide

```python
print("\n=== SELECTION STRATEGY GUIDE ===")
print("""
FILTER METHODS:
  ✓ Fast, scalable
  ✓ Good for initial screening
  ✗ Ignores feature interactions
  ✗ Independent of learning algorithm

WRAPPER METHODS:
  ✓ Considers feature interactions
  ✓ Optimized for specific model
  ✗ Computationally expensive
  ✗ Risk of overfitting

EMBEDDED METHODS:
  ✓ Built into training
  ✓ Considers interactions
  ✓ Efficient
  ✗ Model-specific

PRACTICAL APPROACH:
  1. Start with filter methods (variance, correlation)
  2. Use embedded methods (RF importance, L1)
  3. Validate with wrapper methods if needed
  4. Cross-validate feature selection!
""")
```

## Key Points

- **Variance threshold**: Remove constant/low-variance features
- **Correlation**: With target (useful) or between features (redundant)
- **Chi-square/F-test**: Statistical significance
- **Mutual information**: Captures non-linear relationships
- **RFE**: Iterative elimination using model
- **L1 regularization**: Automatic feature selection
- **Tree importance**: Based on split contributions

## Reflection Questions

1. When would you prefer filter methods over wrapper methods?
2. How does L1 regularization perform feature selection?
3. Why is cross-validation important when doing feature selection?
