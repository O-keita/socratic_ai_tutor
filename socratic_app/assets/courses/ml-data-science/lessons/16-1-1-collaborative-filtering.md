# Collaborative Filtering

## Introduction

Collaborative filtering recommends items based on user-item interactions, leveraging the "wisdom of the crowd." It's the foundation of modern recommendation systems.

## Types of Recommendation Systems

```python
import numpy as np
import pandas as pd

print("=== RECOMMENDATION APPROACHES ===")
print("""
1. CONTENT-BASED:
   - Use item features (genre, description, etc.)
   - "You liked action movies, here's more action movies"
   - Doesn't need other users' data

2. COLLABORATIVE FILTERING:
   - Use user-item interaction patterns
   - "Users similar to you liked this"
   - No item features needed

3. HYBRID:
   - Combine content and collaborative
   - Best of both worlds

Collaborative Filtering types:
  - User-based: Find similar users
  - Item-based: Find similar items
  - Matrix factorization: Latent factors
""")
```

## User-Item Matrix

```python
print("\n=== USER-ITEM MATRIX ===")
print("""
Ratings matrix R (users × items):

         Movie1  Movie2  Movie3  Movie4  Movie5
User1      5       3       ?       1       ?
User2      4       ?       ?       1       ?
User3      ?       1       2       ?       3
User4      1       1       ?       5       4
User5      ?       ?       3       4       4

? = Missing rating (to predict)

Properties:
  - Very SPARSE (most entries missing)
  - Netflix: ~99% sparse
  - Goal: Fill in the missing entries
""")

# Create example matrix
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 0],
    [0, 1, 2, 0, 3],
    [1, 1, 0, 5, 4],
    [0, 0, 3, 4, 4]
])

print("Example ratings (0 = missing):")
print(ratings)
print(f"\nSparsity: {np.sum(ratings == 0) / ratings.size:.1%}")
```

## User-Based Collaborative Filtering

```python
print("\n=== USER-BASED CF ===")
print("""
Idea: Similar users have similar tastes

Algorithm:
1. Find users similar to target user
2. Weight their ratings by similarity
3. Predict based on weighted average

Similarity metrics:
  - Cosine similarity
  - Pearson correlation
  - Adjusted cosine (subtract user mean)
""")

def cosine_similarity(a, b):
    """Cosine similarity between two vectors"""
    # Only consider items rated by both
    mask = (a > 0) & (b > 0)
    if np.sum(mask) == 0:
        return 0
    a_common = a[mask]
    b_common = b[mask]
    return np.dot(a_common, b_common) / (np.linalg.norm(a_common) * np.linalg.norm(b_common) + 1e-8)

def user_based_predict(ratings, user_idx, item_idx, k=2):
    """Predict rating for user on item using k similar users"""
    target_user = ratings[user_idx]
    
    # Calculate similarity with all other users
    similarities = []
    for i, other_user in enumerate(ratings):
        if i != user_idx and ratings[i, item_idx] > 0:  # Must have rated the item
            sim = cosine_similarity(target_user, other_user)
            similarities.append((i, sim, ratings[i, item_idx]))
    
    if not similarities:
        return np.mean(target_user[target_user > 0])  # Fallback to user mean
    
    # Sort by similarity and take top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    
    # Weighted average
    numerator = sum(sim * rating for _, sim, rating in top_k)
    denominator = sum(abs(sim) for _, sim, _ in top_k)
    
    return numerator / (denominator + 1e-8)

# Example: Predict User 0's rating for Movie 2
pred = user_based_predict(ratings, user_idx=0, item_idx=2, k=2)
print(f"Predicted rating for User 0, Movie 2: {pred:.2f}")
```

## Item-Based Collaborative Filtering

```python
print("\n=== ITEM-BASED CF ===")
print("""
Idea: Similar items get similar ratings

Algorithm:
1. Find items similar to target item
2. Weight by similarity to items user has rated
3. Predict based on weighted average

Advantages over user-based:
  - More stable (items don't change)
  - Precompute item similarities
  - Better scalability (fewer items than users)
  
Used by Amazon: "Customers who bought this also bought..."
""")

def item_based_predict(ratings, user_idx, item_idx, k=2):
    """Predict rating using similar items"""
    user_ratings = ratings[user_idx]
    
    # Calculate similarity with all other items
    target_item = ratings[:, item_idx]
    similarities = []
    
    for i in range(ratings.shape[1]):
        if i != item_idx and user_ratings[i] > 0:  # User must have rated this item
            other_item = ratings[:, i]
            sim = cosine_similarity(target_item, other_item)
            similarities.append((i, sim, user_ratings[i]))
    
    if not similarities:
        return np.mean(user_ratings[user_ratings > 0])
    
    # Sort and take top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    
    # Weighted average
    numerator = sum(sim * rating for _, sim, rating in top_k)
    denominator = sum(abs(sim) for _, sim, _ in top_k)
    
    return numerator / (denominator + 1e-8)

pred = item_based_predict(ratings, user_idx=0, item_idx=2, k=2)
print(f"Item-based prediction for User 0, Movie 2: {pred:.2f}")
```

## Matrix Factorization

```python
print("\n=== MATRIX FACTORIZATION ===")
print("""
Idea: Decompose rating matrix into latent factors

R ≈ U × V^T

R: m×n (users × items)
U: m×k (user factors)
V: n×k (item factors)
k: number of latent factors (typically 20-200)

What are latent factors?
  - Hidden characteristics
  - Example: genre preference, production quality, etc.
  - Learned automatically, not predefined

Prediction:
  r̂_ui = u_u · v_i = Σ u_uk × v_ik

Learns:
  - User preferences on each factor
  - Item characteristics on each factor
""")

print("""
Visual:

Users    Factors              Factors    Items
  ┌─┐                           ┌───┐
U1│ │← comedy, action, drama →  │M1 │
U2│ │                           │M2 │
U3│ │     U (5×3)       V^T     │M3 │ (3×5)
U4│ │                   (3×5)   │M4 │
U5│ │                           │M5 │
  └─┘                           └───┘

R = U × V^T = predicted ratings (5×5)
""")
```

## SVD for Recommendations

```python
print("\n=== SVD APPROACH ===")
print("""
SVD decomposes: R = U × Σ × V^T

Problem: Standard SVD doesn't handle missing values

Solution: Iterative optimization

Objective:
  minimize Σ (r_ui - u_u · v_i)² + λ(||U||² + ||V||²)
            (u,i)∈known

Only sum over KNOWN ratings!
λ: regularization to prevent overfitting
""")

def matrix_factorization(R, k=3, steps=100, alpha=0.01, beta=0.01):
    """Simple matrix factorization with SGD"""
    m, n = R.shape
    
    # Initialize random factors
    U = np.random.rand(m, k) * 0.1
    V = np.random.rand(n, k) * 0.1
    
    # Get indices of known ratings
    known = np.where(R > 0)
    
    for step in range(steps):
        for i, j in zip(*known):
            # Prediction error
            error = R[i, j] - np.dot(U[i], V[j])
            
            # Update factors
            U[i] += alpha * (error * V[j] - beta * U[i])
            V[j] += alpha * (error * U[i] - beta * V[j])
        
        # Calculate total error
        if step % 20 == 0:
            total_error = np.sum((R[known] - np.dot(U, V.T)[known])**2)
            print(f"Step {step}: Error = {total_error:.4f}")
    
    return U, V

print("Training matrix factorization...")
U, V = matrix_factorization(ratings, k=3, steps=100)

# Predictions
R_pred = np.dot(U, V.T)
print("\nOriginal (0=missing):")
print(ratings)
print("\nPredicted:")
print(R_pred.round(1))
```

## Handling Cold Start

```python
print("\n=== COLD START PROBLEM ===")
print("""
New users or items have no interaction history!

COLD START SOLUTIONS:

1. NEW USER:
   - Ask for initial preferences
   - Use demographic information
   - Start with popular items
   - Content-based until enough data

2. NEW ITEM:
   - Use item features (content-based)
   - Promote to random users for data
   - Similar items based on metadata

3. HYBRID APPROACHES:
   - Combine CF with content-based
   - Fall back to content when CF fails
""")
```

## Evaluation Metrics

```python
print("\n=== EVALUATION METRICS ===")
print("""
RATING PREDICTION:
  - RMSE: √(Σ(r - r̂)²/n)
  - MAE: Σ|r - r̂|/n
  
RANKING QUALITY:
  - Precision@K: Relevant items in top K
  - Recall@K: Fraction of relevant items found
  - NDCG: Discounted cumulative gain (position matters)
  - MAP: Mean average precision

DIVERSITY:
  - Coverage: % of items recommended
  - Diversity: How different are recommendations?

ONLINE METRICS:
  - Click-through rate (CTR)
  - Conversion rate
  - Watch time
""")

def precision_at_k(recommended, relevant, k):
    """Precision of top-k recommendations"""
    rec_k = set(recommended[:k])
    rel = set(relevant)
    return len(rec_k & rel) / k

def recall_at_k(recommended, relevant, k):
    """Recall of top-k recommendations"""
    rec_k = set(recommended[:k])
    rel = set(relevant)
    return len(rec_k & rel) / len(rel) if rel else 0

# Example
recommended = ['movie1', 'movie2', 'movie3', 'movie4', 'movie5']
relevant = ['movie2', 'movie4', 'movie6']

print(f"Precision@3: {precision_at_k(recommended, relevant, 3):.2f}")
print(f"Recall@3: {recall_at_k(recommended, relevant, 3):.2f}")
print(f"Precision@5: {precision_at_k(recommended, relevant, 5):.2f}")
print(f"Recall@5: {recall_at_k(recommended, relevant, 5):.2f}")
```

## Key Points

- **Collaborative filtering**: Use interaction patterns, not item features
- **User-based**: Find similar users → similar ratings
- **Item-based**: Find similar items → predict from rated items
- **Matrix factorization**: Learn latent factors for users and items
- **Cold start**: Major challenge for new users/items
- **Evaluation**: RMSE for ratings, Precision/Recall for ranking

## Reflection Questions

1. When would item-based CF be preferred over user-based CF?
2. What do the latent factors in matrix factorization represent?
3. How would you handle recommendations for a brand new user?
