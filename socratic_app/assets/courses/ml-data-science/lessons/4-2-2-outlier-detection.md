# Outlier Detection

## Introduction

Outliers are observations that differ significantly from other observations. They can indicate data errors, rare events, or interesting phenomena. Proper identification and handling of outliers is crucial for robust analysis.

## What Are Outliers?

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

print("=== UNDERSTANDING OUTLIERS ===")
print("""
Types of Outliers:

1. POINT OUTLIERS
   - Individual observations that deviate from the distribution
   - Example: Age of 150 in demographic data

2. CONTEXTUAL OUTLIERS
   - Normal in one context, unusual in another
   - Example: 90°F is normal in summer, outlier in winter

3. COLLECTIVE OUTLIERS  
   - Group of observations that are unusual together
   - Example: Sudden spike in network traffic

Causes of Outliers:
  - Data entry errors (typos)
  - Measurement errors (instrument malfunction)
  - Sampling errors (wrong population)
  - Natural variability (genuine rare events)
  - Data processing errors (merging issues)
""")
```

## Statistical Methods

```python
print("\n=== Z-SCORE METHOD ===")

# Generate data with outliers
data = np.concatenate([np.random.normal(50, 10, 100), [120, 5, 130]])
print(f"Data: n={len(data)}, mean={np.mean(data):.2f}, std={np.std(data):.2f}")

# Z-score method
z_scores = np.abs(stats.zscore(data))
threshold = 3
outliers_z = np.where(z_scores > threshold)[0]

print(f"\nZ-score method (|z| > {threshold}):")
print(f"  Outliers found: {len(outliers_z)}")
print(f"  Values: {data[outliers_z]}")
print(f"  Z-scores: {z_scores[outliers_z].round(2)}")

print("""
Z-score interpretation:
  |z| > 2: Unusual (5% of normal data)
  |z| > 2.5: Very unusual (1% of normal data)  
  |z| > 3: Extreme outlier (0.3% of normal data)
  
Limitation: Sensitive to outliers themselves!
""")
```

## IQR Method (Tukey Fences)

```python
print("\n=== IQR METHOD (TUKEY FENCES) ===")

# The IQR method is more robust
data = np.concatenate([np.random.normal(50, 10, 100), [120, 5, 130]])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Define fences
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# Inner fences (mild outliers)
lower_inner = Q1 - 1.5 * IQR
upper_inner = Q3 + 1.5 * IQR

# Outer fences (extreme outliers)
lower_outer = Q1 - 3 * IQR
upper_outer = Q3 + 3 * IQR

print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"\nInner fences (1.5*IQR): [{lower_inner:.2f}, {upper_inner:.2f}]")
print(f"Outer fences (3*IQR): [{lower_outer:.2f}, {upper_outer:.2f}]")

mild_outliers = data[(data < lower_inner) | (data > upper_inner)]
extreme_outliers = data[(data < lower_outer) | (data > upper_outer)]

print(f"\nMild outliers: {mild_outliers}")
print(f"Extreme outliers: {extreme_outliers}")

print("""
Tukey's Rule:
  - Beyond 1.5*IQR: Mild outlier (outside whiskers)
  - Beyond 3*IQR: Extreme outlier
  
Advantage: Not affected by extreme values
""")
```

## Modified Z-Score (MAD)

```python
print("\n=== MODIFIED Z-SCORE (MAD) ===")
print("""
Uses Median Absolute Deviation instead of std.
More robust to outliers than standard z-score.

MAD = median(|x_i - median(x)|)
Modified Z = 0.6745 * (x - median) / MAD
""")

data = np.concatenate([np.random.normal(50, 10, 100), [120, 5, 130]])

median = np.median(data)
mad = np.median(np.abs(data - median))
modified_z = 0.6745 * (data - median) / mad

threshold = 3.5
outliers_mad = np.where(np.abs(modified_z) > threshold)[0]

print(f"Median: {median:.2f}, MAD: {mad:.2f}")
print(f"\nModified z-score method (|z| > {threshold}):")
print(f"  Outliers found: {len(outliers_mad)}")
print(f"  Values: {data[outliers_mad]}")
print(f"  Modified Z-scores: {modified_z[outliers_mad].round(2)}")
```

## Multivariate Outlier Detection

```python
print("\n=== MULTIVARIATE OUTLIERS ===")
print("""
An observation might be normal in each dimension separately,
but unusual when considering all dimensions together.
""")

# Create 2D data with a multivariate outlier
np.random.seed(42)
X = np.random.multivariate_normal([50, 100], [[100, 80], [80, 100]], 100)

# Add an outlier that's not extreme in either dimension alone
X = np.vstack([X, [70, 140]])  # Unusual combination

print("Univariate analysis:")
print(f"  x1 mean={X[:,0].mean():.1f}, outlier={X[-1,0]:.1f}, z={stats.zscore(X[:,0])[-1]:.2f}")
print(f"  x2 mean={X[:,1].mean():.1f}, outlier={X[-1,1]:.1f}, z={stats.zscore(X[:,1])[-1]:.2f}")

# Mahalanobis distance
def mahalanobis_distance(X):
    """Calculate Mahalanobis distance for each point."""
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    cov_inv = np.linalg.inv(cov)
    
    distances = []
    for x in X:
        diff = x - mean
        d = np.sqrt(diff @ cov_inv @ diff)
        distances.append(d)
    return np.array(distances)

distances = mahalanobis_distance(X)
print(f"\nMahalanobis distance of outlier: {distances[-1]:.2f}")
print(f"Mean Mahalanobis distance: {distances[:-1].mean():.2f}")

# Chi-square threshold for 2 dimensions
from scipy.stats import chi2
threshold = chi2.ppf(0.975, df=2)
print(f"Chi-square threshold (95%, df=2): {np.sqrt(threshold):.2f}")
print(f"Is multivariate outlier: {distances[-1] > np.sqrt(threshold)}")
```

## Isolation Forest

```python
print("\n=== ISOLATION FOREST ===")
print("""
Isolation Forest algorithm:
  - Outliers are 'few and different'
  - Easier to isolate from normal points
  - Randomly select feature and split value
  - Outliers need fewer splits to isolate

Advantages:
  - Works well with high-dimensional data
  - No distributional assumptions
  - Efficient (O(n log n))
""")

from sklearn.ensemble import IsolationForest

# Create data with outliers
np.random.seed(42)
X_normal = np.random.randn(200, 2) * 2 + 5
X_outliers = np.random.uniform(-4, 14, (10, 2))
X = np.vstack([X_normal, X_outliers])

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(X)

# -1 for outliers, 1 for inliers
outlier_indices = np.where(predictions == -1)[0]
print(f"Outliers detected: {len(outlier_indices)}")
print(f"Outlier indices (last 10 are known outliers): {outlier_indices}")

# Decision scores (lower = more anomalous)
scores = iso_forest.decision_function(X)
print(f"\nMean score for normal points: {scores[:200].mean():.3f}")
print(f"Mean score for outlier points: {scores[200:].mean():.3f}")
```

## Local Outlier Factor (LOF)

```python
print("\n=== LOCAL OUTLIER FACTOR ===")
print("""
LOF compares local density of a point to its neighbors.
Detects outliers that are local (contextual).

LOF ≈ 1: Similar density to neighbors (normal)
LOF > 1: Lower density than neighbors (outlier)
""")

from sklearn.neighbors import LocalOutlierFactor

# Create clustered data with local outliers
np.random.seed(42)
cluster1 = np.random.randn(100, 2) * 0.5 + [0, 0]
cluster2 = np.random.randn(100, 2) * 2 + [10, 10]
local_outlier = np.array([[3, 3]])  # Between clusters
global_outlier = np.array([[20, 0]])  # Far from everything

X = np.vstack([cluster1, cluster2, local_outlier, global_outlier])

# Fit LOF
lof = LocalOutlierFactor(n_neighbors=20)
predictions = lof.fit_predict(X)

# Get LOF scores (negative, more negative = more outlier-like)
lof_scores = -lof.negative_outlier_factor_

print(f"Mean LOF for cluster 1: {lof_scores[:100].mean():.2f}")
print(f"Mean LOF for cluster 2: {lof_scores[100:200].mean():.2f}")
print(f"LOF for local outlier: {lof_scores[-2]:.2f}")
print(f"LOF for global outlier: {lof_scores[-1]:.2f}")
```

## Handling Outliers

```python
print("\n=== HANDLING OUTLIERS ===")
print("""
1. INVESTIGATE FIRST
   - Is it a data error? Fix or remove.
   - Is it a valid observation? Keep or analyze separately.
   - Is it interesting? May be the most valuable data!

2. REMOVAL
   - Only if clearly erroneous
   - Document why and how many removed
   - Report results with and without

3. TRANSFORMATION
   - Log transform: Reduces impact of large values
   - Winsorization: Cap at percentiles
   - Robust scaling: Use median/IQR instead of mean/std

4. ROBUST METHODS
   - Use median instead of mean
   - Use MAD instead of standard deviation
   - Use robust regression (Huber, RANSAC)

5. SEPARATE ANALYSIS
   - Analyze outliers separately
   - May reveal different phenomenon
""")

# Winsorization example
data = np.concatenate([np.random.normal(50, 10, 100), [120, 5, 130]])

lower_pct = np.percentile(data, 5)
upper_pct = np.percentile(data, 95)
winsorized = np.clip(data, lower_pct, upper_pct)

print("\nWinsorization (5th-95th percentile):")
print(f"  Original range: [{data.min():.2f}, {data.max():.2f}]")
print(f"  Winsorized range: [{winsorized.min():.2f}, {winsorized.max():.2f}]")
print(f"  Original mean: {data.mean():.2f}, std: {data.std():.2f}")
print(f"  Winsorized mean: {winsorized.mean():.2f}, std: {winsorized.std():.2f}")
```

## Key Points

- **Z-score**: Simple but sensitive to outliers
- **IQR method**: Robust, uses percentiles
- **MAD**: Robust alternative to standard deviation
- **Mahalanobis**: Detects multivariate outliers
- **Isolation Forest**: Efficient for high dimensions
- **LOF**: Detects local/contextual outliers
- **Always investigate**: Outliers may be the most interesting data

## Reflection Questions

1. Why might the IQR method be preferred over z-scores for outlier detection?
2. How would you handle outliers differently for EDA versus predictive modeling?
3. When should outliers be kept rather than removed?
