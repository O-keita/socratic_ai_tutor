# K-Means Clustering

## Introduction

K-Means is the most widely used clustering algorithm. It partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids.

## The K-Means Algorithm

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=== K-MEANS ALGORITHM ===")
print("""
Goal: Partition n points into K clusters

ALGORITHM:
  1. Initialize K cluster centroids (randomly)
  2. REPEAT until convergence:
     a. ASSIGN: Each point to nearest centroid
     b. UPDATE: Move centroids to mean of assigned points
  3. Return cluster assignments

Objective: Minimize within-cluster sum of squares (WCSS):
  J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

Where:
  - Cᵢ is cluster i
  - μᵢ is centroid of cluster i
""")
```

## K-Means in Practice

```python
print("\n=== K-MEANS IN PRACTICE ===")

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

print(f"Data shape: {X.shape}")

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

print(f"\nK-Means Results:")
print(f"  Cluster sizes: {np.bincount(y_pred)}")
print(f"  Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"  Number of iterations: {kmeans.n_iter_}")

# Centroids
print(f"\nCluster centroids:")
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: [{centroid[0]:.2f}, {centroid[1]:.2f}]")
```

## Step-by-Step Iteration

```python
print("\n=== STEP-BY-STEP K-MEANS ===")

def kmeans_step_by_step(X, K, max_iter=10, random_state=42):
    """Demonstrate K-Means iterations"""
    np.random.seed(random_state)
    
    # Random initialization
    idx = np.random.choice(len(X), K, replace=False)
    centroids = X[idx].copy()
    
    print(f"Initial centroids (random selection):")
    for i, c in enumerate(centroids):
        print(f"  Centroid {i}: [{c[0]:.2f}, {c[1]:.2f}]")
    
    for iteration in range(max_iter):
        # Step 1: Assign points to nearest centroid
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = distances.argmin(axis=1)
        
        # Step 2: Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        # Check convergence
        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        centroids = new_centroids
        
        print(f"\nIteration {iteration + 1}:")
        print(f"  Cluster sizes: {np.bincount(labels)}")
        print(f"  Max centroid shift: {shift:.4f}")
        
        if shift < 1e-4:
            print("  Converged!")
            break
    
    return labels, centroids

# Run step-by-step
labels, centroids = kmeans_step_by_step(X[:100], K=4, max_iter=5)
```

## Initialization Methods

```python
print("\n=== INITIALIZATION METHODS ===")
print("""
Random initialization can lead to poor results!

METHODS:
1. RANDOM: Pick K random points as initial centroids
   - Simple but inconsistent
   - Run multiple times (n_init)

2. K-MEANS++ (default in sklearn):
   - First centroid: random
   - Subsequent centroids: weighted by distance to nearest existing centroid
   - Spreads out initial centroids
   - Much more consistent results

3. MULTIPLE RUNS (n_init):
   - Run K-Means n_init times with different initializations
   - Keep best result (lowest inertia)
   - sklearn default: n_init=10
""")

# Compare initializations
print("Comparing initialization methods (10 runs each):")

inertias_random = []
inertias_plus = []

for i in range(10):
    km_random = KMeans(n_clusters=4, init='random', n_init=1, random_state=i)
    km_random.fit(X)
    inertias_random.append(km_random.inertia_)
    
    km_plus = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=i)
    km_plus.fit(X)
    inertias_plus.append(km_plus.inertia_)

print(f"\nRandom init inertia: {np.mean(inertias_random):.1f} ± {np.std(inertias_random):.1f}")
print(f"K-Means++ inertia: {np.mean(inertias_plus):.1f} ± {np.std(inertias_plus):.1f}")
print("\nK-Means++ is more consistent and usually finds better solutions.")
```

## Choosing K: The Elbow Method

```python
print("\n=== CHOOSING K: ELBOW METHOD ===")
print("""
Plot inertia (WCSS) vs K:
  - Inertia always decreases as K increases
  - Look for "elbow" - where decrease slows down
  - Elbow suggests natural number of clusters
""")

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

print("K  | Inertia | Decrease")
print("-" * 30)
for i, k in enumerate(K_range):
    decrease = inertias[i-1] - inertias[i] if i > 0 else 0
    print(f"{k:2d} | {inertias[i]:7.1f} | {decrease:7.1f}")

print("""
Look for the "elbow" where the decrease becomes much smaller.
In this case, K=4 is the elbow (we generated 4 clusters!).
""")
```

## Silhouette Analysis

```python
print("\n=== SILHOUETTE ANALYSIS ===")
print("""
Silhouette Score measures cluster quality:

For each point:
  a = average distance to points in same cluster (cohesion)
  b = average distance to points in nearest other cluster (separation)
  s = (b - a) / max(a, b)

Interpretation:
  s ≈ 1: Point is well clustered
  s ≈ 0: Point is on boundary
  s < 0: Point may be in wrong cluster

Overall score: average across all points
""")

from sklearn.metrics import silhouette_score, silhouette_samples

print("Silhouette scores for different K:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"  K={k}: {score:.3f}")

# Detailed silhouette for K=4
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)
sample_scores = silhouette_samples(X, labels)

print(f"\nPer-cluster silhouette scores (K=4):")
for i in range(4):
    cluster_scores = sample_scores[labels == i]
    print(f"  Cluster {i}: mean={cluster_scores.mean():.3f}, min={cluster_scores.min():.3f}")
```

## K-Means Limitations

```python
print("\n=== K-MEANS LIMITATIONS ===")
print("""
1. ASSUMES SPHERICAL CLUSTERS
   - Fails on elongated or non-convex shapes
   
2. ASSUMES SIMILAR CLUSTER SIZES
   - Large clusters can "absorb" small ones
   
3. MUST SPECIFY K IN ADVANCE
   - Not always obvious what K should be
   
4. SENSITIVE TO OUTLIERS
   - Outliers pull centroids toward them
   
5. LOCAL MINIMA
   - May not find global optimum
   - Mitigated by multiple runs (n_init)

6. ONLY HARD ASSIGNMENTS
   - Each point belongs to exactly one cluster
   - No "soft" or probabilistic membership
""")

# Example: K-Means fails on non-spherical data
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_moons = km_moons.fit_predict(X_moons)

print("K-Means on moon-shaped data:")
print(f"  Silhouette score: {silhouette_score(X_moons, labels_moons):.3f}")
print("  (Poor performance - K-Means can't find the true moon shapes!)")
```

## Mini-Batch K-Means

```python
print("\n=== MINI-BATCH K-MEANS ===")
print("""
For large datasets, standard K-Means is slow.
Mini-batch K-Means uses random batches for updates.

Advantages:
  - Much faster (doesn't see all data each iteration)
  - Good approximation of standard K-Means
  
Tradeoffs:
  - Slightly higher inertia
  - Results can vary more
""")

from sklearn.cluster import MiniBatchKMeans
import time

# Large dataset
X_large, _ = make_blobs(n_samples=100000, centers=10, random_state=42)

# Standard K-Means
start = time.time()
km_std = KMeans(n_clusters=10, random_state=42, n_init=1)
km_std.fit(X_large)
time_std = time.time() - start

# Mini-Batch K-Means
start = time.time()
km_mb = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=1, batch_size=1000)
km_mb.fit(X_large)
time_mb = time.time() - start

print(f"Standard K-Means: {time_std:.2f}s, inertia={km_std.inertia_:.0f}")
print(f"Mini-Batch K-Means: {time_mb:.2f}s, inertia={km_mb.inertia_:.0f}")
print(f"Speedup: {time_std/time_mb:.1f}x")
```

## Key Points

- **K-Means**: Partition into K clusters by minimizing within-cluster variance
- **Algorithm**: Iterate assign → update until convergence
- **K-Means++**: Better initialization, more consistent results
- **Elbow method**: Plot inertia vs K to find optimal K
- **Silhouette score**: Evaluate cluster quality (higher is better)
- **Limitations**: Spherical clusters, fixed K, sensitive to outliers
- **Mini-batch**: For large datasets, trades some accuracy for speed

## Reflection Questions

1. Why might K-Means give different results when run multiple times?
2. How would you handle clustering when you don't know the true number of clusters?
3. What pre-processing steps are essential before applying K-Means?
