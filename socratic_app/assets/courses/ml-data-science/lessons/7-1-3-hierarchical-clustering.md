# Hierarchical Clustering

## Introduction

Hierarchical clustering builds a tree of clusters (dendrogram) that shows relationships between clusters at all levels. It provides rich information about data structure without requiring K in advance.

## Agglomerative Clustering

```python
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score

np.random.seed(42)

print("=== HIERARCHICAL CLUSTERING ===")
print("""
Two approaches:
  AGGLOMERATIVE (bottom-up):
    1. Start: each point is its own cluster
    2. Merge: combine two closest clusters
    3. Repeat until one cluster remains
    
  DIVISIVE (top-down):
    1. Start: all points in one cluster
    2. Split: divide cluster into two
    3. Repeat until each point is its own cluster

Agglomerative is much more common and is what sklearn implements.

Result: DENDROGRAM (tree of merges)
""")
```

## Linkage Methods

```python
print("\n=== LINKAGE METHODS ===")
print("""
How do we measure distance between CLUSTERS (not points)?

SINGLE LINKAGE (minimum):
  d(A, B) = min distance between any pair of points
  - Can find elongated clusters
  - Sensitive to noise/outliers (chaining)

COMPLETE LINKAGE (maximum):
  d(A, B) = max distance between any pair of points
  - Produces compact clusters
  - Less sensitive to outliers

AVERAGE LINKAGE:
  d(A, B) = average of all pairwise distances
  - Compromise between single and complete

WARD'S METHOD (default in sklearn):
  d(A, B) = increase in total within-cluster variance if merged
  - Minimizes variance like K-Means
  - Tends to produce similar-sized clusters
  - Most commonly used
""")

# Generate sample data
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)

# Compare linkage methods
linkages = ['ward', 'complete', 'average', 'single']

print("Silhouette scores with different linkage methods (n_clusters=3):")
for link in linkages:
    if link == 'ward':
        agg = AgglomerativeClustering(n_clusters=3, linkage=link)
    else:
        agg = AgglomerativeClustering(n_clusters=3, linkage=link)
    labels = agg.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"  {link:10s}: {score:.3f}")
```

## The Dendrogram

```python
print("\n=== THE DENDROGRAM ===")
print("""
A dendrogram is a tree diagram showing:
  - Vertical axis: Distance at which clusters merge
  - Horizontal axis: Data points or clusters
  - Height of merge: Dissimilarity between merged clusters

Reading a dendrogram:
  - Cut horizontally at any height to get clusters
  - Higher cut = fewer clusters
  - Lower cut = more clusters
  
To choose number of clusters:
  - Look for long vertical lines (large gaps)
  - Cut there to get natural clusters
""")

# Compute linkage matrix
Z = linkage(X, method='ward')

print("Linkage matrix shape:", Z.shape)
print("""
Each row of linkage matrix:
  [cluster1_idx, cluster2_idx, distance, n_points_in_new_cluster]
""")

print("\nFirst 5 merges:")
for i in range(5):
    print(f"  Merge {i+1}: clusters {int(Z[i,0])} and {int(Z[i,1])}, "
          f"distance={Z[i,2]:.3f}, size={int(Z[i,3])}")

print("\nLast 3 merges (forming final clusters):")
for i in range(-3, 0):
    print(f"  Merge: clusters {int(Z[i,0])} and {int(Z[i,1])}, "
          f"distance={Z[i,2]:.3f}, size={int(Z[i,3])}")
```

## Cutting the Dendrogram

```python
print("\n=== CUTTING THE DENDROGRAM ===")
print("""
Two ways to cut:
  1. By number of clusters (n_clusters)
  2. By distance threshold (distance_threshold)
""")

# Cut by number of clusters
from scipy.cluster.hierarchy import fcluster

for n_clusters in [2, 3, 4, 5]:
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    score = silhouette_score(X, labels)
    print(f"  n_clusters={n_clusters}: silhouette={score:.3f}")

# Cut by distance threshold
print("\nCutting by distance threshold:")
distances = Z[:, 2]
print(f"Distance range: [{distances.min():.2f}, {distances.max():.2f}]")

for thresh in [5, 10, 20, 50]:
    labels = fcluster(Z, thresh, criterion='distance')
    n_clusters = len(np.unique(labels))
    if n_clusters > 1:
        score = silhouette_score(X, labels)
        print(f"  threshold={thresh:2d}: {n_clusters} clusters, silhouette={score:.3f}")
    else:
        print(f"  threshold={thresh:2d}: {n_clusters} cluster (all together)")
```

## Agglomerative Clustering in Sklearn

```python
print("\n=== AGGLOMERATIVE CLUSTERING IN SKLEARN ===")

# Standard usage
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

print(f"Cluster assignments: {np.bincount(labels)}")
print(f"Silhouette score: {silhouette_score(X, labels):.3f}")

# With distance threshold (auto-determine n_clusters)
print("\nUsing distance_threshold instead of n_clusters:")
agg_auto = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward')
labels_auto = agg_auto.fit_predict(X)

print(f"Number of clusters found: {len(np.unique(labels_auto))}")
print(f"Cluster sizes: {np.bincount(labels_auto)}")

# Access the tree structure
print(f"\nTree structure:")
print(f"  Number of leaves: {agg_auto.n_leaves_}")
print(f"  Number of connected components: {agg_auto.n_connected_components_}")
```

## Connectivity Constraints

```python
print("\n=== CONNECTIVITY CONSTRAINTS ===")
print("""
Can constrain which points can be merged:
  - Only merge nearby points (spatial constraint)
  - Useful for image segmentation
  - Prevents merging distant clusters

Uses connectivity matrix:
  - connectivity[i,j] = 1 if i and j can be merged
  - Often based on k-nearest neighbors
""")

from sklearn.neighbors import kneighbors_graph

# Create connectivity based on k-NN
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# Cluster with connectivity constraint
agg_conn = AgglomerativeClustering(n_clusters=3, linkage='ward', 
                                    connectivity=connectivity)
labels_conn = agg_conn.fit_predict(X)

print("With connectivity constraint:")
print(f"  Cluster sizes: {np.bincount(labels_conn)}")
print(f"  Silhouette score: {silhouette_score(X, labels_conn):.3f}")
```

## Advantages and Disadvantages

```python
print("\n=== PROS AND CONS ===")
print("""
ADVANTAGES:
  ✓ No need to specify K in advance
  ✓ Dendrogram shows full cluster hierarchy
  ✓ Can cut at any level for different granularities
  ✓ Results are deterministic (no random init)
  ✓ Can find clusters of various shapes (single linkage)
  ✓ Works with any distance metric

DISADVANTAGES:
  ✗ O(n³) time complexity (slow for large datasets)
  ✗ O(n²) space (stores distance matrix)
  ✗ Cannot "undo" merge decisions
  ✗ Sensitive to noise and outliers (especially single linkage)
  ✗ Difficult to interpret with many points

WHEN TO USE:
  - Small to medium datasets (< few thousand points)
  - Want to explore different numbers of clusters
  - Need hierarchical relationships
  - Data has natural hierarchy
""")
```

## Hierarchical vs K-Means

```python
print("\n=== HIERARCHICAL vs K-MEANS ===")
print("""
                    | HIERARCHICAL          | K-MEANS
--------------------+-----------------------+------------------
Specify K           | Optional              | Required
Scalability         | O(n³) poor            | O(nK) good
Reproducibility     | Deterministic         | Depends on init
Output              | Dendrogram            | Flat clusters
Cluster shapes      | Flexible (linkage)    | Spherical
Undo decisions      | No                    | Yes (iterative)
Large datasets      | Difficult             | Works well
""")

# Compare on sample data
from sklearn.cluster import KMeans

print("\nComparison on sample data (3 clusters):")

# K-Means
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = km.fit_predict(X)
score_km = silhouette_score(X, labels_km)

# Hierarchical
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X)
score_agg = silhouette_score(X, labels_agg)

print(f"  K-Means: silhouette={score_km:.3f}")
print(f"  Hierarchical (Ward): silhouette={score_agg:.3f}")
```

## Key Points

- **Hierarchical clustering**: Builds tree of clusters (dendrogram)
- **Agglomerative**: Bottom-up, merge closest clusters
- **Linkage methods**: Single, complete, average, Ward
- **Ward linkage**: Minimizes variance, often best for globular clusters
- **Dendrogram**: Visualize hierarchy, cut at any level
- **No K required**: Can explore multiple cluster numbers
- **Scalability**: O(n³) limits to smaller datasets

## Reflection Questions

1. How would you decide where to cut a dendrogram if there's no clear "gap"?
2. When would single linkage be preferred over Ward linkage?
3. How might you use hierarchical clustering for a dataset too large to fit in memory?
