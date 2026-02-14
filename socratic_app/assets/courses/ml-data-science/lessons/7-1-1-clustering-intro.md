# Introduction to Clustering

## Introduction

Clustering is an unsupervised learning technique that groups similar data points together without labeled examples. It discovers natural structure in data and is fundamental to many applications.

## What is Clustering?

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=== INTRODUCTION TO CLUSTERING ===")
print("""
CLUSTERING:
  - Group similar objects together
  - NO labels provided (unsupervised)
  - Discover natural structure in data

Key Questions:
  - What makes objects "similar"?
  - How many groups (clusters)?
  - What shape are the clusters?

Applications:
  - Customer segmentation
  - Image compression
  - Anomaly detection
  - Document organization
  - Gene expression analysis
  - Social network analysis
""")
```

## Types of Clustering

```python
print("\n=== TYPES OF CLUSTERING ===")
print("""
1. PARTITIONAL CLUSTERING
   - Divide data into K non-overlapping groups
   - Each point belongs to exactly one cluster
   - Examples: K-Means, K-Medoids

2. HIERARCHICAL CLUSTERING
   - Create tree of clusters (dendrogram)
   - Can be agglomerative (bottom-up) or divisive (top-down)
   - Examples: Ward, Complete linkage

3. DENSITY-BASED CLUSTERING
   - Groups based on dense regions
   - Can find arbitrary-shaped clusters
   - Examples: DBSCAN, OPTICS

4. MODEL-BASED CLUSTERING
   - Assume data from mixture of distributions
   - Learn parameters of distributions
   - Examples: Gaussian Mixture Models

5. SPECTRAL CLUSTERING
   - Use eigenvalues of similarity matrix
   - Good for non-convex clusters
""")
```

## Distance Metrics

```python
print("\n=== DISTANCE METRICS ===")
print("""
Clustering depends on how we measure "similarity" (or distance):

EUCLIDEAN DISTANCE:
  d(x, y) = √Σ(xᵢ - yᵢ)²
  - Most common, "straight line" distance
  - Sensitive to feature scales

MANHATTAN DISTANCE:
  d(x, y) = Σ|xᵢ - yᵢ|
  - Sum of absolute differences
  - Less sensitive to outliers

COSINE DISTANCE:
  d(x, y) = 1 - cos(θ) = 1 - (x·y)/(|x||y|)
  - Based on angle between vectors
  - Good for text/high-dimensional sparse data

CORRELATION DISTANCE:
  d(x, y) = 1 - correlation(x, y)
  - Measures pattern similarity
  - Ignores magnitude
""")

# Demonstrate different distances
from scipy.spatial.distance import euclidean, cityblock, cosine

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("Example distances between [1,2,3] and [4,5,6]:")
print(f"  Euclidean: {euclidean(x, y):.3f}")
print(f"  Manhattan: {cityblock(x, y):.3f}")
print(f"  Cosine: {cosine(x, y):.3f}")
```

## Importance of Scaling

```python
print("\n=== IMPORTANCE OF SCALING ===")
print("""
Features on different scales will bias clustering!

Example:
  Age: 20-80 (range 60)
  Income: $20,000-$200,000 (range 180,000)
  
Without scaling, Income dominates distance calculations.

ALWAYS scale before clustering:
  - StandardScaler: z = (x - mean) / std
  - MinMaxScaler: scale to [0, 1]
""")

# Example data
data = pd.DataFrame({
    'Age': [25, 35, 45, 30, 55],
    'Income': [30000, 50000, 75000, 40000, 120000],
    'SpendingScore': [60, 40, 50, 70, 30]
})

print("Original data:")
print(data)

print("\nFeature ranges:")
for col in data.columns:
    print(f"  {col}: [{data[col].min()}, {data[col].max()}], range={data[col].max()-data[col].min()}")

# Scale
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

print("\nAfter StandardScaler (mean=0, std=1):")
print(data_scaled.round(2))
```

## Generating Cluster Data

```python
print("\n=== GENERATING TEST DATA ===")

# Well-separated blobs
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, 
                               cluster_std=0.60, random_state=42)
print(f"Blob data: {X_blobs.shape[0]} samples, {len(np.unique(y_blobs))} true clusters")

# Non-spherical clusters (moons)
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
print(f"Moon data: {X_moons.shape[0]} samples, {len(np.unique(y_moons))} true clusters")

# Real-world-like: customer data
np.random.seed(42)
n_customers = 500

# Three customer segments
segment1 = np.random.randn(n_customers//3, 2) * 0.5 + [2, 5]  # High income, high spend
segment2 = np.random.randn(n_customers//3, 2) * 0.5 + [5, 2]  # High income, low spend
segment3 = np.random.randn(n_customers//3, 2) * 0.5 + [3, 3]  # Medium everything

X_customers = np.vstack([segment1, segment2, segment3])
print(f"Customer data: {X_customers.shape[0]} samples (3 segments)")
```

## Evaluating Clustering

```python
print("\n=== EVALUATING CLUSTERING ===")
print("""
Challenge: No labels to compare against!

INTERNAL METRICS (no ground truth):
  - Silhouette Score: Cohesion vs separation
  - Calinski-Harabasz: Variance ratio
  - Davies-Bouldin: Cluster similarity

EXTERNAL METRICS (when labels available):
  - Adjusted Rand Index
  - Normalized Mutual Information
  - Homogeneity, Completeness, V-measure

VISUAL INSPECTION:
  - Plot clusters in 2D/3D
  - Check for reasonable groupings
  - Domain expert validation
""")

from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Example with blob data
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_blobs)

print("Evaluation metrics for K-Means on blob data:")
print(f"  Silhouette Score: {silhouette_score(X_blobs, labels):.3f}")
print(f"  Calinski-Harabasz: {calinski_harabasz_score(X_blobs, labels):.1f}")

print("""
Silhouette Score interpretation:
  - Range: -1 to 1
  - Higher is better
  - > 0.5: Good separation
  - < 0.25: May be wrong K or overlapping clusters
""")
```

## Common Challenges

```python
print("\n=== COMMON CHALLENGES ===")
print("""
1. CHOOSING NUMBER OF CLUSTERS (K)
   - No "correct" answer often
   - Use elbow method, silhouette analysis
   - Domain knowledge helps

2. NON-SPHERICAL CLUSTERS
   - K-Means assumes spherical clusters
   - Use DBSCAN, spectral clustering for arbitrary shapes

3. VARYING CLUSTER SIZES/DENSITIES
   - Some algorithms struggle with unequal clusters
   - DBSCAN can handle varying densities

4. HIGH DIMENSIONALITY
   - Distance becomes less meaningful
   - Consider dimensionality reduction first

5. OUTLIERS
   - Can distort cluster centers
   - Use robust methods (DBSCAN, K-Medoids)

6. SCALABILITY
   - Large datasets need efficient algorithms
   - Mini-batch K-Means, approximate methods
""")
```

## Key Points

- **Clustering**: Unsupervised grouping of similar objects
- **Distance metrics**: Define what "similar" means (Euclidean, Manhattan, Cosine)
- **Scaling**: Always scale features before clustering
- **Types**: Partitional, hierarchical, density-based, model-based
- **Evaluation**: Internal metrics (silhouette), external (if labels), visual
- **Challenges**: Choosing K, non-spherical clusters, outliers
- **Applications**: Segmentation, anomaly detection, preprocessing

## Reflection Questions

1. How would you decide between using Euclidean distance vs cosine distance?
2. Why might clustering results change dramatically with different feature scales?
3. When would hierarchical clustering be preferred over K-Means?
