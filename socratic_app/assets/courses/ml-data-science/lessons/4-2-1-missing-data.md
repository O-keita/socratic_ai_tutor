# Handling Missing Data

## Introduction

Missing data is ubiquitous in real-world datasets. Understanding why data is missing and how to handle it appropriately is crucial for valid analysis.

## Types of Missing Data

```python
import numpy as np
import pandas as pd

np.random.seed(42)

print("=== TYPES OF MISSING DATA ===")
print("""
1. MCAR (Missing Completely At Random)
   - Missingness is unrelated to any variable
   - Example: Survey form lost randomly in mail
   - Safe to use any imputation method
   
2. MAR (Missing At Random)  
   - Missingness depends on OBSERVED variables
   - Example: Men less likely to report weight
   - Can be handled with proper imputation
   
3. MNAR (Missing Not At Random)
   - Missingness depends on UNOBSERVED values
   - Example: High earners don't report income
   - Most problematic - may cause bias
   - May need specialized methods
""")
```

## Detecting Missing Data

```python
print("\n=== DETECTING MISSING DATA ===")

# Create sample data with missing values
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 35, 40, 28, 55],
    'income': [50000, np.nan, 60000, 80000, np.nan, np.nan, 45000, 70000, 55000, 90000],
    'education': ['BS', 'MS', 'BS', np.nan, 'PhD', 'BS', 'MS', np.nan, 'BS', 'PhD'],
    'rating': [4.5, 3.8, np.nan, 4.2, np.nan, 3.9, 4.1, 4.0, np.nan, 4.8]
})

print("Sample DataFrame:")
print(df)

print("\n=== CHECKING FOR MISSING VALUES ===")
print("\nMissing count per column:")
print(df.isnull().sum())

print("\nMissing percentage per column:")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\nTotal cells with missing values:")
print(f"  {df.isnull().sum().sum()} out of {df.size} ({df.isnull().sum().sum()/df.size*100:.1f}%)")

print("\nRows with any missing value:")
print(f"  {df.isnull().any(axis=1).sum()} out of {len(df)}")

print("\nComplete cases (no missing):")
print(f"  {df.dropna().shape[0]} out of {len(df)}")
```

## Visualizing Missing Patterns

```python
print("\n=== MISSING DATA PATTERNS ===")

# Create larger dataset with patterns
np.random.seed(42)
n = 100
df_large = pd.DataFrame({
    'var_a': np.random.normal(50, 10, n),
    'var_b': np.random.normal(100, 20, n),
    'var_c': np.random.choice(['X', 'Y', 'Z'], n),
    'var_d': np.random.uniform(0, 1, n),
    'var_e': np.random.exponential(10, n)
})

# Introduce different missing patterns
# MCAR - random 10%
mcar_mask = np.random.random(n) < 0.1
df_large.loc[mcar_mask, 'var_a'] = np.nan

# MAR - missing var_b when var_c == 'Z'
df_large.loc[df_large['var_c'] == 'Z', 'var_b'] = np.nan

# High var_e tends to be missing (MNAR-like)
df_large.loc[df_large['var_e'] > df_large['var_e'].quantile(0.8), 'var_e'] = np.nan

print("Missing Pattern Summary:")
print(df_large.isnull().sum())

# Correlation of missingness
missing_matrix = df_large.isnull().astype(int)
print("\nMissingness Correlation:")
print(missing_matrix.corr().round(3))
```

## Deletion Methods

```python
print("\n=== DELETION METHODS ===")

# Original data
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5, np.nan, 7],
    'B': [10, np.nan, 30, 40, 50, 60, 70],
    'C': [100, 200, 300, np.nan, 500, 600, 700]
})
print("Original Data:")
print(df)

# Listwise deletion (complete case analysis)
print("\nListwise Deletion (dropna):")
print(df.dropna())
print(f"  Rows remaining: {len(df.dropna())} of {len(df)}")

# Drop rows with all missing
print("\nDrop rows where ALL values are missing:")
df_test = df.copy()
df_test.loc[len(df_test)] = [np.nan, np.nan, np.nan]  # Add all-missing row
print(df_test.dropna(how='all'))

# Drop columns with too many missing
print("\nDrop columns with >30% missing:")
threshold = 0.7 * len(df)
print(df.dropna(axis=1, thresh=int(threshold)))

# Pairwise deletion
print("\nPairwise Deletion (for correlation):")
print(f"  Correlation A-B: {df['A'].corr(df['B']):.3f} (uses {df[['A','B']].dropna().shape[0]} pairs)")
print(f"  Correlation A-C: {df['A'].corr(df['C']):.3f} (uses {df[['A','C']].dropna().shape[0]} pairs)")
```

## Simple Imputation

```python
print("\n=== SIMPLE IMPUTATION ===")

df = pd.DataFrame({
    'numeric': [10, 20, np.nan, 40, 50, np.nan, 70, 80],
    'category': ['A', 'B', np.nan, 'A', 'B', 'A', np.nan, 'B'],
    'skewed': [100, 200, np.nan, 150, 5000, np.nan, 180, 220]
})
print("Original:")
print(df)

# Mean imputation
df_mean = df.copy()
df_mean['numeric'] = df_mean['numeric'].fillna(df_mean['numeric'].mean())
print(f"\nMean Imputation (numeric): {df['numeric'].mean():.2f}")
print(df_mean['numeric'].values)

# Median imputation (better for skewed)
df_median = df.copy()
df_median['skewed'] = df_median['skewed'].fillna(df_median['skewed'].median())
print(f"\nMedian Imputation (skewed): {df['skewed'].median():.2f}")
print(df_median['skewed'].values)

# Mode imputation (categorical)
df_mode = df.copy()
mode_value = df_mode['category'].mode()[0]
df_mode['category'] = df_mode['category'].fillna(mode_value)
print(f"\nMode Imputation (category): {mode_value}")
print(df_mode['category'].values)

# Forward/backward fill
df_ffill = df.copy()
df_ffill['numeric'] = df_ffill['numeric'].fillna(method='ffill')
print("\nForward Fill (numeric):")
print(df_ffill['numeric'].values)
```

## Problems with Simple Imputation

```python
print("\n=== PROBLEMS WITH MEAN IMPUTATION ===")

# Demonstrate variance reduction
np.random.seed(42)
original = np.random.normal(100, 20, 100)
# Make 20% missing
missing_idx = np.random.choice(100, 20, replace=False)
with_missing = original.copy()
with_missing[missing_idx] = np.nan

# Mean impute
mean_imputed = with_missing.copy()
mean_imputed[np.isnan(mean_imputed)] = np.nanmean(with_missing)

print(f"Original variance: {np.var(original):.2f}")
print(f"After mean imputation: {np.var(mean_imputed):.2f}")
print(f"Variance reduced by: {(1 - np.var(mean_imputed)/np.var(original))*100:.1f}%")

print("""
Problems with mean/median imputation:
  1. Reduces variance (artificially)
  2. Distorts correlations
  3. Underestimates standard errors
  4. Biases statistical tests
  5. Ignores relationships between variables
""")
```

## Advanced Imputation

```python
print("\n=== ADVANCED IMPUTATION ===")
print("""
1. REGRESSION IMPUTATION
   - Predict missing values from other variables
   - Preserves relationships
   - Still underestimates variance

2. STOCHASTIC REGRESSION
   - Add random noise to regression predictions
   - Better preserves variance

3. K-NEAREST NEIGHBORS (KNN)
   - Use similar observations to impute
   - Good for mixed data types

4. MULTIPLE IMPUTATION
   - Create multiple imputed datasets
   - Analyze each, combine results
   - Properly accounts for uncertainty
   - Gold standard for inference

5. ITERATIVE IMPUTATION (MICE)
   - Multiple Imputation by Chained Equations
   - Each variable imputed in round-robin fashion
""")

# Simple KNN example
from sklearn.impute import KNNImputer

df_numeric = pd.DataFrame({
    'x': [1, 2, np.nan, 4, 5, 6, np.nan, 8],
    'y': [10, 20, 30, np.nan, 50, 60, 70, 80],
    'z': [100, 200, 300, 400, 500, np.nan, 700, 800]
})
print("\nKNN Imputation:")
print("Before:")
print(df_numeric)

imputer = KNNImputer(n_neighbors=2)
imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
print("\nAfter (K=2):")
print(imputed)
```

## Imputation Strategies by Data Type

```python
print("\n=== IMPUTATION STRATEGIES ===")
print("""
NUMERIC DATA:
  - Mean: Normal distribution, MCAR
  - Median: Skewed data, outliers present
  - KNN: Use similar observations
  - Regression: Preserve relationships

CATEGORICAL DATA:
  - Mode: Most common category
  - "Missing" category: Treat as its own class
  - Model-based: Predict category

TIME SERIES:
  - Forward fill: Last known value
  - Backward fill: Next known value  
  - Interpolation: Linear/spline between points
  - Seasonal: Use same period last cycle

PRACTICAL GUIDELINES:
  1. Understand WHY data is missing first
  2. <5% missing: Most methods work
  3. 5-20% missing: Use multiple imputation
  4. >20% missing: Consider dropping variable
  5. Always compare results with/without imputation
""")
```

## Key Points

- **MCAR, MAR, MNAR**: Understanding mechanism determines strategy
- **Listwise deletion**: Simple but loses data
- **Mean imputation**: Reduces variance, distorts relationships
- **KNN imputation**: Uses similar observations
- **Multiple imputation**: Gold standard for inference
- **Always document**: Report missing data handling choices

## Reflection Questions

1. How would you determine if data is missing at random or not?
2. Why does mean imputation underestimate variance?
3. When might it be appropriate to simply delete observations with missing data?
