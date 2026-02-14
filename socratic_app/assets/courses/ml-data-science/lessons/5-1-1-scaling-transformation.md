# Feature Scaling and Transformation

## Introduction

Feature scaling and transformation are essential preprocessing steps that normalize data ranges and improve algorithm performance. Many machine learning algorithms are sensitive to the scale of input features.

## Why Scale Features?

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

np.random.seed(42)

print("=== WHY SCALE FEATURES? ===")
print("""
Problem: Features have different scales

Example:
  - Age: 0-100
  - Income: 0-1,000,000
  - Rating: 1-5

Without scaling:
  - Distance-based algorithms dominated by large-scale features
  - Gradient descent converges slowly
  - Regularization affects features unequally

Algorithms REQUIRING scaling:
  - K-Nearest Neighbors (distance-based)
  - Support Vector Machines (distance-based)
  - Neural Networks (gradient descent)
  - Principal Component Analysis
  - K-Means Clustering
  - Logistic/Linear Regression with regularization

Algorithms NOT requiring scaling:
  - Decision Trees
  - Random Forests
  - Gradient Boosting
  - Naive Bayes
""")

# Demonstrate the problem
df = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'income': [30000, 50000, 70000, 90000, 110000],
    'rating': [3.5, 4.0, 3.8, 4.2, 4.5]
})
print("\nUnscaled data:")
print(df)
print(f"\nVariances: age={df['age'].var():.0f}, income={df['income'].var():.0f}, rating={df['rating'].var():.2f}")
```

## Standard Scaling (Z-Score Normalization)

```python
print("\n=== STANDARD SCALING ===")
print("""
Formula: z = (x - mean) / std

Properties:
  - Mean = 0, Std = 1
  - Preserves shape of distribution
  - Works well with normally distributed data
  - Sensitive to outliers

When to use:
  - Data is approximately normal
  - Algorithm assumes normal distribution
  - Using regularization
""")

# Create sample data
X = np.array([
    [25, 30000, 3.5],
    [35, 50000, 4.0],
    [45, 70000, 3.8],
    [55, 90000, 4.2],
    [65, 110000, 4.5]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original data:")
print(X)
print("\nStandard scaled:")
print(X_scaled.round(3))
print(f"\nMeans: {X_scaled.mean(axis=0).round(6)}")
print(f"Stds: {X_scaled.std(axis=0).round(6)}")

# Manual calculation
print("\nManual verification for first column (age):")
mean = X[:, 0].mean()
std = X[:, 0].std()
manual_scaled = (X[:, 0] - mean) / std
print(f"  Mean: {mean}, Std: {std:.2f}")
print(f"  Scaled: {manual_scaled.round(3)}")
```

## Min-Max Scaling

```python
print("\n=== MIN-MAX SCALING ===")
print("""
Formula: x_scaled = (x - min) / (max - min)

Properties:
  - Range: [0, 1] (or custom range)
  - Preserves relationships and shape
  - Sensitive to outliers
  
When to use:
  - Need bounded features (e.g., image pixels)
  - Algorithm expects input in [0, 1] range
  - No strong outliers
""")

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

print("Min-Max scaled [0, 1]:")
print(X_minmax.round(3))
print(f"\nMins: {X_minmax.min(axis=0)}")
print(f"Maxs: {X_minmax.max(axis=0)}")

# Custom range [-1, 1]
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
X_custom = scaler_custom.fit_transform(X)
print("\nMin-Max scaled [-1, 1]:")
print(X_custom.round(3))
```

## Robust Scaling

```python
print("\n=== ROBUST SCALING ===")
print("""
Formula: x_scaled = (x - median) / IQR

Properties:
  - Uses median and IQR (robust statistics)
  - Less sensitive to outliers
  - Doesn't guarantee bounded range
  
When to use:
  - Data has outliers
  - Don't want to remove outliers
  - Need robust scaling
""")

# Data with outlier
X_outlier = np.array([
    [25, 30000],
    [35, 50000],
    [45, 70000],
    [55, 90000],
    [200, 500000]  # Outlier
])

print("Data with outlier:")
print(X_outlier)

# Compare scalers
print("\nStandard Scaling (affected by outlier):")
ss = StandardScaler()
print(ss.fit_transform(X_outlier).round(3))

print("\nRobust Scaling (less affected):")
rs = RobustScaler()
print(rs.fit_transform(X_outlier).round(3))
```

## Log and Power Transformations

```python
print("\n=== LOG TRANSFORMATION ===")
print("""
Use when:
  - Data is right-skewed (long right tail)
  - Multiplicative relationships
  - Variance increases with mean

Common variants:
  - log(x): Natural log
  - log10(x): Log base 10
  - log1p(x): log(1 + x) for data with zeros
""")

# Skewed data
skewed = np.array([100, 200, 300, 500, 1000, 2000, 5000, 10000])
print(f"Original (skewed): {skewed}")
print(f"  Mean: {skewed.mean():.0f}, Median: {np.median(skewed):.0f}")

log_transformed = np.log(skewed)
print(f"\nLog transformed: {log_transformed.round(2)}")
print(f"  Mean: {log_transformed.mean():.2f}, Median: {np.median(log_transformed):.2f}")

print("\n=== POWER TRANSFORMATIONS ===")
print("""
Box-Cox Transformation:
  - Finds optimal power transformation
  - Requires strictly positive data
  
Yeo-Johnson Transformation:
  - Works with zero and negative values
""")

from sklearn.preprocessing import PowerTransformer

# Box-Cox (positive data only)
X_skewed = np.random.exponential(10, 100).reshape(-1, 1)
pt_boxcox = PowerTransformer(method='box-cox')
X_boxcox = pt_boxcox.fit_transform(X_skewed)

# Yeo-Johnson (any data)
pt_yj = PowerTransformer(method='yeo-johnson')
X_yj = pt_yj.fit_transform(X_skewed)

print(f"Original skewness: {pd.Series(X_skewed.flatten()).skew():.2f}")
print(f"After Box-Cox: {pd.Series(X_boxcox.flatten()).skew():.2f}")
print(f"After Yeo-Johnson: {pd.Series(X_yj.flatten()).skew():.2f}")
```

## Quantile Transformation

```python
print("\n=== QUANTILE TRANSFORMATION ===")
print("""
Transforms data to follow a specified distribution
(uniform or normal).

Properties:
  - Non-linear transformation
  - Reduces impact of outliers
  - Forces desired distribution shape

Useful when:
  - Highly skewed data
  - Many outliers
  - Need normal distribution
""")

from sklearn.preprocessing import QuantileTransformer

# Highly skewed data
X_very_skewed = np.random.exponential(1, 1000).reshape(-1, 1)

# Transform to uniform distribution
qt_uniform = QuantileTransformer(output_distribution='uniform', random_state=42)
X_uniform = qt_uniform.fit_transform(X_very_skewed)

# Transform to normal distribution
qt_normal = QuantileTransformer(output_distribution='normal', random_state=42)
X_normal = qt_normal.fit_transform(X_very_skewed)

print("Original:")
print(f"  Min: {X_very_skewed.min():.2f}, Max: {X_very_skewed.max():.2f}")
print(f"  Skewness: {pd.Series(X_very_skewed.flatten()).skew():.2f}")

print("\nUniform transformed:")
print(f"  Min: {X_uniform.min():.2f}, Max: {X_uniform.max():.2f}")

print("\nNormal transformed:")
print(f"  Mean: {X_normal.mean():.2f}, Std: {X_normal.std():.2f}")
print(f"  Skewness: {pd.Series(X_normal.flatten()).skew():.2f}")
```

## Choosing the Right Scaler

```python
print("\n=== SCALER SELECTION GUIDE ===")
print("""
STANDARD SCALER:
  ✓ Data is normally distributed
  ✓ No significant outliers
  ✓ Using regularization
  
MIN-MAX SCALER:
  ✓ Need bounded range [0, 1]
  ✓ Neural networks (some architectures)
  ✓ Image data
  
ROBUST SCALER:
  ✓ Data has outliers
  ✓ Don't want outliers to dominate
  
LOG TRANSFORM:
  ✓ Right-skewed data
  ✓ Multiplicative relationships
  ✓ Count data
  
POWER TRANSFORM:
  ✓ Need to normalize heavily skewed data
  ✓ Want automated parameter selection
  
QUANTILE TRANSFORM:
  ✓ Extreme outliers
  ✓ Need specific distribution shape

IMPORTANT: Fit scaler on training data only!
           Transform test data with same parameters.
""")
```

## Key Points

- **Standard scaling**: Centers to mean=0, std=1
- **Min-Max scaling**: Bounds to [0, 1] range
- **Robust scaling**: Uses median/IQR, handles outliers
- **Log transform**: Reduces right skewness
- **Power transforms**: Automated normalization
- **Fit on training only**: Prevent data leakage
- **Some algorithms don't need scaling**: Tree-based methods

## Reflection Questions

1. Why is it important to fit the scaler on training data only?
2. When would Robust Scaling be preferred over Standard Scaling?
3. How does feature scaling affect the interpretation of model coefficients?
