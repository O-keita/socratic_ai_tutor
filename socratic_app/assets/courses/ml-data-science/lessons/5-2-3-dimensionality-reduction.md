# Dimensionality Reduction

## Introduction

Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation while preserving important information. This helps with visualization, computation, and avoiding the curse of dimensionality.

## Why Reduce Dimensions?

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_classification

np.random.seed(42)

print("=== WHY DIMENSIONALITY REDUCTION? ===")
print("""
Problems with High Dimensionality:

1. CURSE OF DIMENSIONALITY
   - Data becomes sparse in high dimensions
   - Distance metrics become less meaningful
   - Need exponentially more data

2. COMPUTATIONAL COST
   - More features = slower training
   - Memory limitations

3. OVERFITTING
   - More features can lead to memorizing noise
   - Especially with limited samples

4. VISUALIZATION
   - Can only visualize 2-3 dimensions directly

Solutions:
  - Feature Selection: Keep subset of original features
  - Dimensionality Reduction: Create new, fewer features
    (combinations of original features)
""")
```

## Principal Component Analysis (PCA)

```python
print("\n=== PRINCIPAL COMPONENT ANALYSIS (PCA) ===")
print("""
PCA finds directions of maximum variance:
  - First PC: Direction of maximum variance
  - Second PC: Orthogonal, next maximum variance
  - And so on...

Properties:
  - Linear transformation
  - Components are uncorrelated
  - Preserves as much variance as possible

Use when:
  - Data is roughly linear
  - Variance indicates importance
  - Need interpretable components
""")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load digits dataset (64 features)
digits = load_digits()
X = digits.data
y = digits.target

print(f"Original shape: {X.shape}")

# Always scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Full PCA to see variance explained
pca_full = PCA()
pca_full.fit(X_scaled)

print("\nVariance explained by each component:")
var_ratio = pca_full.explained_variance_ratio_
cumulative = np.cumsum(var_ratio)
print(f"  First 10 components: {cumulative[9]:.2%}")
print(f"  First 20 components: {cumulative[19]:.2%}")
print(f"  First 30 components: {cumulative[29]:.2%}")

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
print(f"\nReduced to 2D: {X_2d.shape}")
print(f"Variance retained: {sum(pca_2d.explained_variance_ratio_):.2%}")
```

## Choosing Number of Components

```python
print("\n=== CHOOSING N_COMPONENTS ===")
print("""
Methods to choose number of components:

1. CUMULATIVE VARIANCE THRESHOLD
   - Keep components until 90-95% variance explained

2. ELBOW METHOD
   - Plot variance vs components
   - Look for "elbow" where gains diminish

3. RECONSTRUCTION ERROR
   - Minimize error in reconstruction

4. DOMAIN KNOWLEDGE
   - Based on downstream task needs
""")

# Method 1: Variance threshold
target_variance = 0.95
n_components_95 = np.argmax(cumulative >= target_variance) + 1
print(f"Components for 95% variance: {n_components_95}")

# Method 2: Automatic with variance ratio
pca_auto = PCA(n_components=0.95)  # Keep 95% variance
X_auto = pca_auto.fit_transform(X_scaled)
print(f"Shape with 95% variance: {X_auto.shape}")

# Show component importance
print("\nTop 10 components by variance explained:")
for i in range(10):
    print(f"  PC{i+1}: {var_ratio[i]:.2%} (cumulative: {cumulative[i]:.2%})")
```

## Interpreting PCA Components

```python
print("\n=== INTERPRETING PCA ===")
print("""
Each component is a linear combination of original features.
Loadings show contribution of each feature.
""")

# Simple example with interpretable features
np.random.seed(42)
X_simple = pd.DataFrame({
    'height': np.random.normal(170, 10, 100),
    'weight': np.random.normal(70, 15, 100),
    'age': np.random.normal(35, 10, 100),
    'income': np.random.normal(50000, 15000, 100)
})
# Add correlations
X_simple['weight'] = X_simple['weight'] + 0.5 * (X_simple['height'] - 170)
X_simple['income'] = X_simple['income'] + 500 * X_simple['age']

X_scaled_simple = StandardScaler().fit_transform(X_simple)
pca_simple = PCA()
pca_simple.fit(X_scaled_simple)

print("PCA Loadings:")
loadings = pd.DataFrame(
    pca_simple.components_.T,
    columns=[f'PC{i+1}' for i in range(4)],
    index=X_simple.columns
)
print(loadings.round(3))

print("\nInterpretation:")
print("  PC1: Contrast between height/weight and age/income")
print("  PC2: General 'size' (all positive loadings)")
```

## t-SNE for Visualization

```python
print("\n=== t-SNE (t-Distributed Stochastic Neighbor Embedding) ===")
print("""
t-SNE is primarily for visualization:
  - Non-linear dimensionality reduction
  - Preserves local structure (clusters)
  - Good for revealing clusters/groups

Properties:
  - Non-parametric (no transform for new data)
  - Stochastic (different runs give different results)
  - Computationally intensive
  - Perplexity parameter controls local/global balance
""")

from sklearn.manifold import TSNE

# Apply t-SNE to digits
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

print(f"t-SNE output shape: {X_tsne.shape}")
print(f"Note: t-SNE has no transform() method for new data!")

print("""
t-SNE Tips:
  - Use PCA first to reduce to ~50 dimensions (speed)
  - Try different perplexity values (5-50)
  - Don't interpret distances between clusters
  - Mainly useful for visualization
""")
```

## UMAP

```python
print("\n=== UMAP (Uniform Manifold Approximation) ===")
print("""
UMAP is similar to t-SNE but:
  - Faster, scales better
  - Better preserves global structure
  - Has transform() for new data
  - More parameters to tune

Key parameters:
  - n_neighbors: Local vs global structure (like perplexity)
  - min_dist: How tightly points are packed
""")

# UMAP example (conceptual - requires umap-learn package)
print("""
from umap import UMAP

umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
X_umap = umap_model.fit_transform(X)

# UMAP can transform new data!
X_new_umap = umap_model.transform(X_new)
""")
```

## Linear Discriminant Analysis (LDA)

```python
print("\n=== LINEAR DISCRIMINANT ANALYSIS (LDA) ===")
print("""
LDA is supervised dimensionality reduction:
  - Uses class labels to find directions
  - Maximizes between-class variance
  - Minimizes within-class variance
  - At most (n_classes - 1) components

Difference from PCA:
  - PCA: Unsupervised (ignores labels)
  - LDA: Supervised (uses labels)
  - LDA better for classification tasks
""")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA on digits (10 classes -> max 9 components)
lda = LinearDiscriminantAnalysis(n_components=9)
X_lda = lda.fit_transform(X, y)

print(f"Original shape: {X.shape}")
print(f"LDA shape: {X_lda.shape}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_[:5].round(3)}")

# 2D for visualization
lda_2d = LinearDiscriminantAnalysis(n_components=2)
X_lda_2d = lda_2d.fit_transform(X, y)
print(f"\n2D LDA variance explained: {sum(lda_2d.explained_variance_ratio_):.2%}")
```

## Comparison of Methods

```python
print("\n=== METHOD COMPARISON ===")
print("""
METHOD          SUPERVISED   LINEAR   FAST   NEW DATA   USE CASE
─────────────────────────────────────────────────────────────────
PCA             No           Yes      Yes    Yes        General reduction
LDA             Yes          Yes      Yes    Yes        Classification
t-SNE           No           No       No     No         Visualization
UMAP            No           No       Yes    Yes        Visualization + speed

CHOOSING A METHOD:

PCA:
  ✓ First choice for general reduction
  ✓ Fast and scalable
  ✓ Components are interpretable
  ✓ Good before t-SNE/UMAP

LDA:
  ✓ Classification task
  ✓ Want to use class information
  ✓ Need interpretable directions

t-SNE:
  ✓ Pure visualization
  ✓ Reveal cluster structure
  ✗ Don't use for feature engineering

UMAP:
  ✓ Visualization with speed
  ✓ Need to transform new data
  ✓ Better global structure than t-SNE
""")
```

## Practical Pipeline

```python
print("\n=== PRACTICAL PIPELINE ===")
print("""
1. Start with PCA
   - Reduce to 50-100 dimensions
   - Preserves structure for downstream tasks

2. For Visualization
   - Apply t-SNE or UMAP to PCA output
   - Much faster than on original data

3. For Classification
   - Consider LDA after PCA
   - Uses label information
""")

# Example pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# This is common: Scale → PCA → Model
print("Example Pipeline: StandardScaler → PCA → LogisticRegression")
```

## Key Points

- **PCA**: Linear, unsupervised, preserves variance
- **LDA**: Linear, supervised, maximizes class separation
- **t-SNE**: Non-linear, for visualization only
- **UMAP**: Non-linear, faster, can transform new data
- **Scale first**: Always standardize before PCA
- **Choose components**: By variance threshold or downstream performance
- **Combine methods**: PCA first, then t-SNE/UMAP for visualization

## Reflection Questions

1. Why is scaling important before applying PCA?
2. When would you choose LDA over PCA?
3. Why shouldn't t-SNE be used for feature engineering?
