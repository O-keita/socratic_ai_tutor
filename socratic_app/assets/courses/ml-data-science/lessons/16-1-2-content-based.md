# Content-Based Filtering

## Introduction

Content-based filtering recommends items similar to what a user has liked before, using item features rather than other users' behavior.

## Content-Based vs Collaborative

```python
import numpy as np
import pandas as pd

print("=== CONTENT-BASED APPROACH ===")
print("""
CONTENT-BASED FILTERING:

Uses ITEM FEATURES to make recommendations

Example: Movie recommendation
  User liked: "The Matrix" (action, sci-fi, thriller)
  Recommend: Other action/sci-fi movies

Advantages:
✓ No cold start for new items (if features known)
✓ Works for individual users
✓ Recommendations are explainable
✓ No need for other users' data

Disadvantages:
✗ Limited to feature similarity
✗ Hard to capture subtle preferences
✗ Over-specialization (filter bubble)
✗ Requires good features
""")
```

## Item Feature Representation

```python
print("\n=== FEATURE REPRESENTATION ===")
print("""
How to represent items:

1. CATEGORICAL FEATURES:
   Genre: [1, 0, 1, 0, 0]  # action=1, comedy=0, drama=1, ...
   Director: one-hot or embedding

2. TEXT FEATURES (TF-IDF):
   Description → TF-IDF vector
   "A thrilling action movie..." → [0.2, 0.0, 0.3, ...]

3. EMBEDDINGS:
   Learn dense vectors from data
   Pre-trained (Word2Vec) or learned
""")

# Example: Movie features
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Toy Story', 'The Avengers', 'Finding Nemo'],
    'action': [1, 1, 0, 1, 0],
    'sci_fi': [1, 1, 0, 0, 0],
    'animation': [0, 0, 1, 0, 1],
    'comedy': [0, 0, 1, 0, 1],
    'thriller': [1, 1, 0, 0, 0]
})

print("Movie features (genre encoding):")
print(movies.to_string(index=False))

# Feature vectors
feature_cols = ['action', 'sci_fi', 'animation', 'comedy', 'thriller']
movie_vectors = movies[feature_cols].values
print(f"\nFeature matrix shape: {movie_vectors.shape}")
```

## Computing Item Similarity

```python
print("\n=== ITEM SIMILARITY ===")
print("""
Similarity measures for feature vectors:

1. COSINE SIMILARITY:
   sim(a, b) = (a · b) / (||a|| × ||b||)
   Range: [-1, 1] or [0, 1] for non-negative

2. EUCLIDEAN DISTANCE:
   dist(a, b) = √(Σ(a_i - b_i)²)
   Similarity = 1 / (1 + dist)

3. JACCARD SIMILARITY (for binary):
   sim(a, b) = |a ∩ b| / |a ∪ b|
""")

def cosine_similarity_matrix(vectors):
    """Compute pairwise cosine similarity"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-8)
    return np.dot(normalized, normalized.T)

sim_matrix = cosine_similarity_matrix(movie_vectors)

print("Item similarity matrix:")
sim_df = pd.DataFrame(sim_matrix, 
                       index=movies['title'], 
                       columns=movies['title'])
print(sim_df.round(2))
```

## User Profile Building

```python
print("\n=== USER PROFILE ===")
print("""
Build user preference profile from their history:

Simple approach: Weighted average of liked items

user_profile = Σ (rating_i × item_vector_i) / Σ rating_i

Example:
  User rated:
    - "The Matrix": 5 stars
    - "Inception": 4 stars
    - "Toy Story": 2 stars
    
  Profile = weighted average of these feature vectors
  → Higher weights on action/sci-fi
""")

def build_user_profile(item_vectors, user_ratings):
    """Build user profile from rated items"""
    weighted_sum = np.zeros(item_vectors.shape[1])
    total_weight = 0
    
    for item_idx, rating in user_ratings.items():
        weighted_sum += rating * item_vectors[item_idx]
        total_weight += rating
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return weighted_sum

# User ratings (item_index: rating)
user_ratings = {0: 5.0, 1: 4.0, 2: 2.0}  # Matrix=5, Inception=4, Toy Story=2

user_profile = build_user_profile(movie_vectors, user_ratings)
print("User profile vector:")
for i, feat in enumerate(feature_cols):
    print(f"  {feat}: {user_profile[i]:.2f}")
```

## Making Recommendations

```python
print("\n=== GENERATING RECOMMENDATIONS ===")
print("""
Score unrated items by similarity to user profile:

score(item) = cosine_similarity(user_profile, item_vector)

Then rank items by score.
""")

def recommend_content_based(user_profile, item_vectors, rated_items, top_k=3):
    """Recommend items based on user profile"""
    scores = []
    
    for idx in range(len(item_vectors)):
        if idx not in rated_items:
            # Cosine similarity
            sim = np.dot(user_profile, item_vectors[idx])
            sim /= (np.linalg.norm(user_profile) * np.linalg.norm(item_vectors[idx]) + 1e-8)
            scores.append((idx, sim))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

recommendations = recommend_content_based(user_profile, movie_vectors, 
                                          set(user_ratings.keys()))

print("Recommendations:")
for idx, score in recommendations:
    print(f"  {movies['title'][idx]}: score = {score:.3f}")
```

## TF-IDF for Text Features

```python
print("\n=== TF-IDF FOR TEXT ===")
print("""
For text-based content (descriptions, reviews):

from sklearn.feature_extraction.text import TfidfVectorizer

descriptions = [
    "A computer hacker learns about the true nature of reality",
    "A thief enters people's dreams to steal secrets",
    "Toys come to life when humans aren't looking",
    "Earth's mightiest heroes must come together",
    "A clownfish searches for his kidnapped son"
]

# Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = tfidf.fit_transform(descriptions)

# Now can compute similarity on text
text_similarity = cosine_similarity(tfidf_matrix)
""")

print("""
TF-IDF captures:
  - Important words in each document
  - Down-weights common words
  - Creates sparse feature vectors
  
For better results:
  - Use word embeddings (Word2Vec, BERT)
  - Capture semantic similarity
  - "king" similar to "monarch"
""")
```

## Hybrid Approaches

```python
print("\n=== HYBRID RECOMMENDERS ===")
print("""
Combine content-based and collaborative filtering:

1. WEIGHTED HYBRID:
   score = α × content_score + (1-α) × cf_score
   
2. SWITCHING:
   Use CF when enough data
   Fall back to content for cold start
   
3. FEATURE AUGMENTATION:
   Use CF predictions as features
   Or use content similarity in CF
   
4. META-LEVEL:
   Content-based model's output → CF input
   Or vice versa

Benefits:
  - Handles cold start better
  - Combines strengths of both
  - More robust recommendations
""")

def hybrid_recommend(user_id, item_vectors, cf_model, alpha=0.5):
    """Hybrid recommendation combining content and CF"""
    scores = []
    
    for item_idx in range(len(item_vectors)):
        # Content-based score
        content_score = content_similarity(user_profile, item_vectors[item_idx])
        
        # Collaborative filtering score
        cf_score = cf_model.predict(user_id, item_idx)
        
        # Weighted combination
        hybrid_score = alpha * content_score + (1 - alpha) * cf_score
        scores.append((item_idx, hybrid_score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)

print("Hybrid approach is used by most production systems.")
```

## Handling the Filter Bubble

```python
print("\n=== DIVERSITY AND EXPLORATION ===")
print("""
FILTER BUBBLE PROBLEM:
  Content-based → more of the same
  User only sees similar items
  No discovery of new interests

SOLUTIONS:

1. DIVERSITY BOOSTING:
   Add diversity penalty to similar items
   Ensure recommendations are varied
   
2. EXPLORATION/EXPLOITATION:
   Occasionally show random items
   Multi-armed bandit approach
   
3. SERENDIPITY:
   Recommend occasionally surprising items
   Items different but potentially liked

4. EXPLICIT DIVERSITY:
   Max-Marginal-Relevance (MMR)
   score = λ × relevance - (1-λ) × max_similarity_to_selected
""")

def mmr_rerank(candidates, selected, item_vectors, lambda_param=0.5):
    """Max-Marginal-Relevance for diversity"""
    if not selected:
        return candidates[0]  # Return best
    
    best_score = float('-inf')
    best_item = None
    
    for idx, relevance in candidates:
        if idx in [s[0] for s in selected]:
            continue
        
        # Max similarity to already selected items
        max_sim = max(
            np.dot(item_vectors[idx], item_vectors[s[0]]) 
            for s in selected
        )
        
        # MMR score
        mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
        
        if mmr_score > best_score:
            best_score = mmr_score
            best_item = (idx, relevance)
    
    return best_item

print("MMR balances relevance with diversity.")
```

## Key Points

- **Content-based**: Recommend similar items using features
- **User profile**: Weighted average of liked item vectors
- **Item similarity**: Cosine similarity on feature vectors
- **TF-IDF**: Convert text to numerical features
- **Hybrid**: Combine content + collaborative for best results
- **Diversity**: Avoid filter bubbles with exploration

## Reflection Questions

1. What are the trade-offs between content-based and collaborative filtering?
2. How can embeddings improve content-based recommendations?
3. Why is diversity important in recommendation systems?
