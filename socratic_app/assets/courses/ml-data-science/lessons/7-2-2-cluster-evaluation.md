# Cluster Evaluation Metrics

## Introduction

Evaluating clustering is challenging because we often don't have ground truth labels. This lesson covers both internal metrics (no labels needed) and external metrics (when labels available).

## The Evaluation Challenge

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score,
                             completeness_score, v_measure_score)

np.random.seed(42)

print("=== CLUSTER EVALUATION CHALLENGE ===")
print("""
Unlike supervised learning, clustering has no clear target!

Questions we need to answer:
  - How well are clusters separated?
  - How cohesive (compact) are clusters?
  - Is K correct?
  - Which algorithm/parameters work best?

Two types of metrics:
  1. INTERNAL: Based only on data and cluster assignments
  2. EXTERNAL: Compare to known ground truth labels
""")
```

## Internal Metrics: Silhouette Score

```python
print("\n=== SILHOUETTE SCORE ===")
print("""
Measures how similar points are to their own cluster vs other clusters.

For each point i:
  a(i) = mean distance to other points in same cluster (cohesion)
  b(i) = mean distance to points in nearest other cluster (separation)
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

Overall score = mean of all s(i)

Interpretation:
  s ≈ +1: Well clustered
  s ≈  0: On boundary between clusters
  s ≈ -1: Probably in wrong cluster

Range: [-1, 1], higher is better
""")

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Cluster with different K values
print("Silhouette score for different K:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"  K={k}: {score:.3f}")

print("\nHighest at K=4 (we generated 4 clusters!)")
```

## Per-Cluster Silhouette Analysis

```python
print("\n=== PER-CLUSTER SILHOUETTE ===")

from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)

# Get silhouette for each sample
sample_silhouettes = silhouette_samples(X, labels)

print("Per-cluster silhouette statistics:")
for i in range(4):
    cluster_silhouettes = sample_silhouettes[labels == i]
    print(f"  Cluster {i}:")
    print(f"    Size: {len(cluster_silhouettes)}")
    print(f"    Mean: {cluster_silhouettes.mean():.3f}")
    print(f"    Min: {cluster_silhouettes.min():.3f}")
    print(f"    % negative: {(cluster_silhouettes < 0).mean()*100:.1f}%")

print("""
Points with negative silhouette may be misassigned.
Uneven cluster silhouettes may indicate poor clustering.
""")
```

## Internal Metrics: Calinski-Harabasz Index

```python
print("\n=== CALINSKI-HARABASZ INDEX ===")
print("""
Also called Variance Ratio Criterion.

Measures ratio of:
  - Between-cluster dispersion (separation)
  - Within-cluster dispersion (compactness)

CH = (BSS / (k-1)) / (WSS / (n-k))

Where:
  BSS = between-cluster sum of squares
  WSS = within-cluster sum of squares
  k = number of clusters
  n = number of points

Higher is better (no upper bound).
""")

print("Calinski-Harabasz for different K:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = calinski_harabasz_score(X, labels)
    print(f"  K={k}: {score:.1f}")
```

## Internal Metrics: Davies-Bouldin Index

```python
print("\n=== DAVIES-BOULDIN INDEX ===")
print("""
Measures average similarity between each cluster and its most similar cluster.

For clusters i and j:
  R_ij = (s_i + s_j) / d_ij
  
Where:
  s_i = average distance from points in i to centroid of i
  d_ij = distance between centroids i and j

DB = (1/k) * Σ max_j(R_ij)

Lower is better (minimum is 0).
""")

print("Davies-Bouldin for different K:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    print(f"  K={k}: {score:.3f}")

print("\nLowest (best) at K=4")
```

## Comparing Internal Metrics

```python
print("\n=== COMPARING INTERNAL METRICS ===")

print("Summary of internal metrics:")
print(f"{'K':>3} | {'Silhouette':>11} | {'CH Index':>10} | {'DB Index':>10}")
print("-" * 50)

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    print(f"{k:>3} | {sil:>11.3f} | {ch:>10.1f} | {db:>10.3f}")

print("""
Notes:
  - Silhouette: Higher is better, range [-1, 1]
  - Calinski-Harabasz: Higher is better, no upper bound
  - Davies-Bouldin: Lower is better, minimum 0
  
All suggest K=4 is optimal for this data.
""")
```

## External Metrics: Adjusted Rand Index

```python
print("\n=== EXTERNAL METRICS (WITH GROUND TRUTH) ===")
print("""
When we have true labels, we can evaluate clustering quality directly.

ADJUSTED RAND INDEX (ARI):
  - Measures agreement between two clusterings
  - Adjusted for chance
  - Range: [-1, 1], 1 = perfect, 0 = random, <0 = worse than random
""")

print("ARI for different algorithms:")
algorithms = {
    'K-Means (K=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
    'K-Means (K=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
    'K-Means (K=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
    'Hierarchical': AgglomerativeClustering(n_clusters=4),
}

for name, algo in algorithms.items():
    labels = algo.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    print(f"  {name}: {ari:.3f}")
```

## External Metrics: Normalized Mutual Information

```python
print("\n=== NORMALIZED MUTUAL INFORMATION ===")
print("""
Based on information theory:
  - Measures mutual information between clusterings
  - Normalized to [0, 1]
  - 1 = perfect agreement
  
Variants:
  - 'arithmetic': NMI = 2*I(U,V) / (H(U) + H(V))
  - 'geometric': NMI = I(U,V) / sqrt(H(U)*H(V))
""")

for name, algo in algorithms.items():
    labels = algo.fit_predict(X)
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"  {name}: {nmi:.3f}")
```

## Homogeneity, Completeness, V-Measure

```python
print("\n=== HOMOGENEITY, COMPLETENESS, V-MEASURE ===")
print("""
HOMOGENEITY:
  - Each cluster contains only members of a single class
  - "All cluster members are from same class"

COMPLETENESS:
  - All members of a given class are assigned to same cluster
  - "All class members are in same cluster"

V-MEASURE:
  - Harmonic mean of homogeneity and completeness
  - V = 2 * (h * c) / (h + c)

All range [0, 1], 1 is perfect.
""")

print(f"{'Algorithm':>20} | {'Homo':>6} | {'Comp':>6} | {'V-meas':>8}")
print("-" * 50)

for name, algo in algorithms.items():
    labels = algo.fit_predict(X)
    h = homogeneity_score(y_true, labels)
    c = completeness_score(y_true, labels)
    v = v_measure_score(y_true, labels)
    print(f"{name:>20} | {h:>6.3f} | {c:>6.3f} | {v:>8.3f}")
```

## When to Use Which Metric

```python
print("\n=== CHOOSING METRICS ===")
print("""
NO GROUND TRUTH (most common):
  - Silhouette: Easy to interpret, works with any algorithm
  - Calinski-Harabasz: Fast to compute
  - Davies-Bouldin: Good for comparing same K
  
  Recommendation: Use multiple metrics, look for consensus

WITH GROUND TRUTH (validation/research):
  - ARI: Robust, handles different K
  - NMI: Information-theoretic interpretation
  - V-measure: Understand homogeneity vs completeness

SPECIAL CASES:
  - DBSCAN: Exclude noise points from metrics
  - High dimensions: Silhouette can be misleading
  - Large datasets: Sampling may be needed

CAVEATS:
  - Internal metrics favor certain shapes (e.g., spherical)
  - No single metric is perfect
  - Domain knowledge is essential
""")
```

## Practical Workflow

```python
print("\n=== PRACTICAL WORKFLOW ===")
print("""
1. PREPROCESSING
   - Scale features (StandardScaler)
   - Handle outliers if needed

2. EXPLORE K RANGE
   - Try K = 2 to sqrt(n)
   - Plot elbow curve (inertia/WSS)
   - Plot silhouette scores

3. COMPARE ALGORITHMS
   - K-Means, Hierarchical, DBSCAN, GMM
   - Use same evaluation metrics

4. VALIDATE RESULTS
   - Visualize clusters (PCA if needed)
   - Check cluster profiles
   - Domain expert review

5. FINAL SELECTION
   - Choose based on:
     * Best metrics
     * Interpretability
     * Business requirements
""")

# Example comparison
print("\nExample: Comparing algorithms (K=4):")
results = []

km = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
hc = AgglomerativeClustering(n_clusters=4).fit_predict(X)

for name, labels in [('K-Means', km), ('Hierarchical', hc)]:
    results.append({
        'Algorithm': name,
        'Silhouette': silhouette_score(X, labels),
        'ARI': adjusted_rand_score(y_true, labels),
        'NMI': normalized_mutual_info_score(y_true, labels)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

## Key Points

- **Internal metrics**: No labels needed (silhouette, CH, DB)
- **External metrics**: Need ground truth (ARI, NMI, V-measure)
- **Silhouette**: Most widely used, range [-1, 1]
- **ARI**: Adjusted for chance, good for different K
- **No perfect metric**: Use multiple, look for consensus
- **Visualization**: Always supplement metrics with plots
- **Domain knowledge**: Essential for final validation

## Reflection Questions

1. Why might internal metrics prefer spherical clusters even when the true clusters are non-spherical?
2. When would you prioritize homogeneity over completeness?
3. How would you evaluate a clustering when you expect some points to be noise/outliers?
