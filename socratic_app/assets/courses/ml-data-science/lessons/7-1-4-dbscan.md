# DBSCAN: Density-Based Clustering

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters based on dense regions separated by sparse regions. It can discover clusters of arbitrary shapes and automatically identifies outliers.

## The DBSCAN Algorithm

```python
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(42)

print("=== DBSCAN ALGORITHM ===")
print("""
Key Concepts:
  ε (epsilon): Neighborhood radius
  minPts: Minimum points to form dense region

Point Types:
  CORE POINT: Has ≥ minPts within ε radius
  BORDER POINT: Within ε of a core point, but < minPts neighbors
  NOISE POINT: Neither core nor border (outlier)

Algorithm:
  1. For each point, count neighbors within ε
  2. Identify core points (≥ minPts neighbors)
  3. Form clusters from connected core points
  4. Assign border points to nearest cluster
  5. Remaining points are noise

Clusters = dense regions separated by sparse regions
""")
```

## DBSCAN in Practice

```python
print("\n=== DBSCAN IN PRACTICE ===")

# Generate blob data
X_blobs, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_blobs)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"DBSCAN Results (eps=0.5, min_samples=5):")
print(f"  Number of clusters: {n_clusters}")
print(f"  Number of noise points: {n_noise} ({n_noise/len(X_scaled)*100:.1f}%)")

# Cluster sizes (excluding noise)
unique_labels = set(labels) - {-1}
for label in unique_labels:
    count = np.sum(labels == label)
    print(f"  Cluster {label}: {count} points")
```

## Core, Border, and Noise Points

```python
print("\n=== POINT TYPES ===")

# Identify point types
core_mask = np.zeros(len(X_scaled), dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

n_core = np.sum(core_mask)
n_border = np.sum((~core_mask) & (labels != -1))
n_noise = np.sum(labels == -1)

print(f"Point breakdown:")
print(f"  Core points: {n_core} ({n_core/len(X_scaled)*100:.1f}%)")
print(f"  Border points: {n_border} ({n_border/len(X_scaled)*100:.1f}%)")
print(f"  Noise points: {n_noise} ({n_noise/len(X_scaled)*100:.1f}%)")

print("""
Core points:
  - Have enough neighbors to define density
  - Form the "backbone" of clusters
  
Border points:
  - Near core points but don't have enough neighbors
  - On the edge of clusters
  
Noise points:
  - Too far from any core point
  - Labeled as -1 in sklearn
""")
```

## Choosing Parameters: eps and min_samples

```python
print("\n=== CHOOSING PARAMETERS ===")
print("""
eps (epsilon):
  - Too small: Many noise points, fragmented clusters
  - Too large: Clusters merge together
  - Guideline: Use k-distance plot

min_samples:
  - Too small: Many small clusters, noise becomes clusters
  - Too large: Fewer, larger clusters, more noise
  - Guideline: At least dimensions + 1, or 2 * dimensions
""")

# Effect of eps
print("\nEffect of eps (min_samples=5):")
print(f"{'eps':>6} | Clusters | Noise")
print("-" * 30)
for eps in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"{eps:>6.1f} | {n_clusters:>8} | {n_noise:>5}")

# Effect of min_samples
print("\nEffect of min_samples (eps=0.5):")
print(f"{'minPts':>6} | Clusters | Noise")
print("-" * 30)
for min_pts in [2, 3, 5, 10, 15, 20]:
    db = DBSCAN(eps=0.5, min_samples=min_pts)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"{min_pts:>6} | {n_clusters:>8} | {n_noise:>5}")
```

## K-Distance Plot for Choosing eps

```python
print("\n=== K-DISTANCE PLOT ===")
print("""
To choose eps:
  1. Calculate k-nearest neighbor distance for each point
  2. Sort distances in ascending order
  3. Plot sorted distances
  4. Look for "elbow" - sharp increase indicates eps value

k should be min_samples - 1
""")

from sklearn.neighbors import NearestNeighbors

# Calculate k-distances
k = 4  # min_samples - 1
nn = NearestNeighbors(n_neighbors=k+1)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)

# k-th nearest neighbor distance (excluding self)
k_distances = distances[:, k]
k_distances_sorted = np.sort(k_distances)

print(f"k-distance statistics (k={k}):")
print(f"  Min: {k_distances_sorted[0]:.3f}")
print(f"  25%: {np.percentile(k_distances_sorted, 25):.3f}")
print(f"  50%: {np.percentile(k_distances_sorted, 50):.3f}")
print(f"  75%: {np.percentile(k_distances_sorted, 75):.3f}")
print(f"  90%: {np.percentile(k_distances_sorted, 90):.3f}")
print(f"  Max: {k_distances_sorted[-1]:.3f}")

print("""
The elbow point in the k-distance plot suggests eps ≈ 0.5
Points below the elbow are "reachable", above are likely noise.
""")
```

## DBSCAN on Non-Spherical Clusters

```python
print("\n=== DBSCAN ON NON-SPHERICAL CLUSTERS ===")

# Generate moon-shaped data
X_moons, y_moons = make_moons(n_samples=200, noise=0.08, random_state=42)

# Compare K-Means and DBSCAN
from sklearn.cluster import KMeans

# K-Means
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km.fit_predict(X_moons)

# DBSCAN
db = DBSCAN(eps=0.25, min_samples=5)
labels_db = db.fit_predict(X_moons)

# Evaluate
# Note: silhouette score not perfect for non-convex clusters
score_km = silhouette_score(X_moons, labels_km)

# For DBSCAN, exclude noise points for silhouette
non_noise = labels_db != -1
if np.sum(non_noise) > 1 and len(set(labels_db[non_noise])) > 1:
    score_db = silhouette_score(X_moons[non_noise], labels_db[non_noise])
else:
    score_db = np.nan

print("Moon-shaped data comparison:")
print(f"  K-Means: silhouette={score_km:.3f}")
print(f"  DBSCAN: silhouette={score_db:.3f} (excluding {np.sum(~non_noise)} noise points)")

# Check cluster assignments match true labels
from sklearn.metrics import adjusted_rand_score
ari_km = adjusted_rand_score(y_moons, labels_km)
ari_db = adjusted_rand_score(y_moons, labels_db)

print(f"\nAdjusted Rand Index (vs true labels):")
print(f"  K-Means: {ari_km:.3f}")
print(f"  DBSCAN: {ari_db:.3f}")
print("\nDBSCAN correctly identifies the moon shapes!")
```

## Handling Varying Densities

```python
print("\n=== VARYING DENSITIES ===")
print("""
DBSCAN Challenge:
  - Uses single eps for all clusters
  - Struggles when clusters have different densities
  
Solutions:
  1. OPTICS: Ordering Points To Identify Clustering Structure
     - Creates reachability plot
     - Can find clusters at multiple density levels
     
  2. HDBSCAN: Hierarchical DBSCAN
     - Extends DBSCAN with hierarchy
     - Handles varying densities better
""")

# Generate data with varying densities
np.random.seed(42)
X_dense = np.random.randn(200, 2) * 0.3 + [0, 0]
X_sparse = np.random.randn(100, 2) * 1.0 + [5, 0]
X_varying = np.vstack([X_dense, X_sparse])

# DBSCAN with single eps
print("Clusters with varying densities:")
for eps in [0.3, 0.5, 0.8, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_varying)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise")

print("""
No single eps works well:
  - Small eps: Sparse cluster fragments
  - Large eps: Dense cluster absorbs sparse
""")
```

## Advantages and Disadvantages

```python
print("\n=== PROS AND CONS ===")
print("""
ADVANTAGES:
  ✓ Finds arbitrary-shaped clusters
  ✓ Automatically identifies outliers/noise
  ✓ No need to specify number of clusters
  ✓ Robust to outliers (noise points)
  ✓ Works well with spatial data

DISADVANTAGES:
  ✗ Sensitive to eps and min_samples
  ✗ Struggles with varying densities
  ✗ Not suitable for high-dimensional data
  ✗ Doesn't work well if clusters are close
  ✗ Border points can be arbitrary

WHEN TO USE:
  - Non-spherical clusters expected
  - Outlier detection needed
  - Spatial/geographic data
  - Number of clusters unknown
  
WHEN NOT TO USE:
  - High-dimensional data
  - Varying density clusters
  - Need consistent cluster assignments
""")
```

## Key Points

- **DBSCAN**: Density-based clustering, finds arbitrary shapes
- **Parameters**: eps (radius), min_samples (minimum points)
- **Point types**: Core, border, noise
- **Noise detection**: Automatic outlier identification (label -1)
- **K-distance plot**: Helps choose eps parameter
- **Non-spherical**: Excellent for non-globular clusters
- **Limitations**: Single density, parameter sensitive

## Reflection Questions

1. Why might DBSCAN fail when clusters have very different densities?
2. How would you decide between using DBSCAN vs K-Means for a new dataset?
3. What happens to DBSCAN's performance as dimensionality increases?
