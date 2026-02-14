# Handling Missing Values in Features

## Introduction

Missing values are common in real-world datasets and require careful handling during feature engineering. The strategy depends on the missingness mechanism and the model being used.

## Understanding Missingness

```python
import numpy as np
import pandas as pd

np.random.seed(42)

print("=== TYPES OF MISSING DATA ===")
print("""
1. MCAR (Missing Completely At Random)
   - Missingness is random, unrelated to any variable
   - Example: Data lost due to random equipment failure
   
2. MAR (Missing At Random)
   - Missingness depends on OBSERVED variables
   - Example: Young people less likely to report income
   
3. MNAR (Missing Not At Random)
   - Missingness depends on the MISSING value itself
   - Example: High earners hide income

Detection approaches:
  - Compare distributions: missing vs non-missing groups
  - Test if missingness correlates with other features
  - Domain knowledge about data collection
""")
```

## Analyzing Missing Data Patterns

```python
print("\n=== MISSING DATA ANALYSIS ===")

# Create sample data with various missing patterns
n = 1000
df = pd.DataFrame({
    'age': np.random.normal(40, 12, n),
    'income': np.random.exponential(50000, n),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
    'score': np.random.normal(100, 15, n)
})

# Introduce missingness
# MCAR: Random 5% missing in age
df.loc[np.random.choice(n, 50, replace=False), 'age'] = np.nan

# MAR: Income missing more often for younger people
young_mask = df['age'] < 30
df.loc[young_mask & (np.random.random(n) < 0.3), 'income'] = np.nan

# MNAR: High scores tend to be missing
high_score = df['score'] > 115
df.loc[high_score & (np.random.random(n) < 0.4), 'score'] = np.nan

print("Missing Data Summary:")
print(df.isnull().sum())
print(f"\nMissing percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Missing pattern analysis
print("\nMissing Patterns:")
missing_patterns = df.isnull().sum(axis=1).value_counts().sort_index()
print(f"Rows with 0 missing: {missing_patterns.get(0, 0)}")
print(f"Rows with 1 missing: {missing_patterns.get(1, 0)}")
print(f"Rows with 2+ missing: {sum([v for k,v in missing_patterns.items() if k >= 2])}")
```

## Simple Imputation Strategies

```python
print("\n=== SIMPLE IMPUTATION ===")

from sklearn.impute import SimpleImputer

# Create sample data
df = pd.DataFrame({
    'numeric': [1, 2, np.nan, 4, 5, np.nan, 7],
    'category': ['A', 'B', np.nan, 'A', 'B', 'A', np.nan]
})
print("Original:")
print(df)

# Mean imputation for numeric
mean_imputer = SimpleImputer(strategy='mean')
df['numeric_mean'] = mean_imputer.fit_transform(df[['numeric']])

# Median imputation (better for skewed)
median_imputer = SimpleImputer(strategy='median')
df['numeric_median'] = median_imputer.fit_transform(df[['numeric']])

# Mode imputation for categorical
mode_imputer = SimpleImputer(strategy='most_frequent')
df['category_mode'] = mode_imputer.fit_transform(df[['category']].values.reshape(-1, 1))

print("\nAfter imputation:")
print(df)

print("""
Simple Imputation Issues:
  - Reduces variance (artificially)
  - Distorts correlations
  - Doesn't account for uncertainty
""")
```

## KNN Imputation

```python
print("\n=== KNN IMPUTATION ===")
print("""
Uses K-nearest neighbors to impute.
Finds similar rows based on non-missing features.

Advantages:
  - Uses relationships between features
  - Can capture complex patterns
  
Disadvantages:
  - Computationally expensive
  - Sensitive to k and scaling
  - Requires similar observations
""")

from sklearn.impute import KNNImputer

# Sample data
df_knn = pd.DataFrame({
    'feature1': [1, 2, np.nan, 4, 5, 6, np.nan, 8],
    'feature2': [10, 20, 30, np.nan, 50, 60, 70, 80],
    'feature3': [100, 200, 300, 400, 500, 600, 700, 800]
})
print("Original:")
print(df_knn)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=2)
df_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df_knn),
    columns=df_knn.columns
)
print("\nKNN Imputed (k=2):")
print(df_imputed.round(2))
```

## Iterative Imputation (MICE)

```python
print("\n=== ITERATIVE IMPUTATION (MICE) ===")
print("""
Multiple Imputation by Chained Equations:
  1. Fill missing with simple imputation
  2. For each feature, predict missing using other features
  3. Repeat until convergence

Advantages:
  - Models relationships between variables
  - Can handle complex patterns
  - Theoretically sound
""")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Sample data
np.random.seed(42)
X = np.random.randn(100, 4)
# Add some missingness
mask = np.random.random(X.shape) < 0.1
X[mask] = np.nan

print(f"Missing values: {np.isnan(X).sum()}")

# Iterative imputation
iter_imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = iter_imputer.fit_transform(X)

print(f"After imputation: {np.isnan(X_imputed).sum()} missing")
print("\nSample (first 5 rows):")
print("Original:")
print(np.round(X[:5], 2))
print("\nImputed:")
print(np.round(X_imputed[:5], 2))
```

## Creating Missing Indicators

```python
print("\n=== MISSING INDICATORS ===")
print("""
Create binary features indicating missingness.

Why use them:
  - Missingness itself may be informative
  - Preserves information about data quality
  - Model can learn from missingness patterns

When to use:
  - MAR or MNAR data
  - Missingness has predictive value
  - Combined with imputation
""")

df = pd.DataFrame({
    'value': [100, np.nan, 300, np.nan, 500],
    'other': [10, 20, 30, 40, 50]
})

# Add missing indicator
df['value_missing'] = df['value'].isnull().astype(int)

# Impute the original
df['value_imputed'] = df['value'].fillna(df['value'].mean())

print("With missing indicator:")
print(df)

print("""
This preserves both:
  1. Imputed value (for algorithms that need complete data)
  2. Information that value was missing
""")
```

## Imputation for Categorical Variables

```python
print("\n=== CATEGORICAL IMPUTATION ===")

df_cat = pd.DataFrame({
    'category': ['A', 'B', np.nan, 'A', 'C', np.nan, 'B'],
    'numeric': [100, 200, 150, 180, 250, 120, 220]
})
print("Original:")
print(df_cat)

# Method 1: Mode imputation
mode_val = df_cat['category'].mode()[0]
df_cat['cat_mode'] = df_cat['category'].fillna(mode_val)
print(f"\nMode imputation (mode='{mode_val}'):")
print(df_cat)

# Method 2: Treat missing as category
df_cat['cat_missing_class'] = df_cat['category'].fillna('MISSING')
print("\nMissing as category:")
print(df_cat)

# Method 3: Probabilistic imputation
print("\nProbabilistic imputation:")
probs = df_cat['category'].dropna().value_counts(normalize=True)
print(f"Category probabilities: {probs.to_dict()}")

missing_mask = df_cat['category'].isnull()
n_missing = missing_mask.sum()
df_cat.loc[missing_mask, 'cat_prob'] = np.random.choice(
    probs.index, size=n_missing, p=probs.values
)
print(df_cat)
```

## Best Practices

```python
print("\n=== IMPUTATION BEST PRACTICES ===")
print("""
1. UNDERSTAND THE MISSINGNESS
   - Is it MCAR, MAR, or MNAR?
   - What caused the missing data?

2. CHECK PATTERNS
   - How much is missing?
   - Are there patterns in missingness?

3. CHOOSE APPROPRIATE METHOD
   - Simple (mean/median): Low missingness, MCAR
   - KNN/Iterative: MAR, relationships matter
   - Missing indicator: Missingness is informative

4. PRESERVE UNCERTAINTY
   - Consider multiple imputation
   - Use missing indicators

5. VALIDATE
   - Compare distributions before/after
   - Check model performance with/without imputation

6. PREVENT DATA LEAKAGE
   - Fit imputer on training data only
   - Transform test data with same parameters

7. DOCUMENT CHOICES
   - Record what was imputed and how
   - Report missing data statistics
""")

# Example pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("\n=== EXAMPLE PIPELINE ===")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X = np.array([[1, 2], [np.nan, 3], [7, 6]])
X_transformed = pipeline.fit_transform(X)
print("Pipeline: Impute → Scale")
print(f"Input:\n{X}")
print(f"Output:\n{X_transformed.round(2)}")
```

## Key Points

- **MCAR, MAR, MNAR**: Understand why data is missing
- **Simple imputation**: Mean/median for quick solutions
- **KNN/Iterative**: Use relationships between features
- **Missing indicators**: Preserve missingness information
- **Categorical**: Mode, "MISSING" class, or probabilistic
- **Fit on training only**: Prevent data leakage
- **Document everything**: Track imputation decisions

## Reflection Questions

1. How would you determine if missing data is MCAR vs MAR?
2. When is treating "missing" as its own category appropriate?
3. Why should imputers be fit only on training data?
