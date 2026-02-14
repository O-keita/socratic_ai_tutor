# Word Embeddings

## Introduction

Word embeddings are dense vector representations that capture semantic meaning. Unlike sparse BoW vectors, embeddings place similar words close together in continuous vector space.

## From Sparse to Dense

```python
import numpy as np
import pandas as pd

print("=== WORD EMBEDDINGS ===")
print("""
SPARSE (BoW/TF-IDF):
  - One dimension per word in vocabulary
  - Most values are zero
  - "king" → [0,0,0,1,0,0,...,0] (10,000+ dimensions)
  - No semantic similarity captured

DENSE (Embeddings):
  - Fixed lower dimensions (100-300 typical)
  - All values are non-zero floats
  - "king" → [0.2, -0.4, 0.7, ..., 0.1] (300 dimensions)
  - Similar words have similar vectors

Key insight: Words appearing in similar CONTEXTS have similar MEANINGS
""")
```

## The Distributional Hypothesis

```python
print("\n=== DISTRIBUTIONAL HYPOTHESIS ===")
print("""
"You shall know a word by the company it keeps" - J.R. Firth

Words with similar meanings appear in similar contexts:
  "I ate an APPLE for breakfast"
  "I ate an ORANGE for breakfast"
  
  "The DOG barked loudly"
  "The CAT meowed softly"

Word embedding algorithms learn from this co-occurrence!

Methods:
  - Word2Vec (Google, 2013)
  - GloVe (Stanford, 2014)
  - FastText (Facebook, 2016)
""")
```

## Word2Vec: Skip-gram and CBOW

```python
print("\n=== WORD2VEC ===")
print("""
Two architectures:

SKIP-GRAM:
  - Predict context words from center word
  - Input: "king"
  - Output: ["the", "was", "crowned"]
  - Better for rare words
  
CBOW (Continuous Bag of Words):
  - Predict center word from context
  - Input: ["the", "was", "crowned"]
  - Output: "king"
  - Faster training

Both use neural network with:
  - Input layer: One-hot word vector
  - Hidden layer: Embedding (what we want!)
  - Output layer: Softmax over vocabulary

Training creates embeddings as side effect!
""")
```

## Using Pre-trained Embeddings

```python
print("\n=== USING PRE-TRAINED EMBEDDINGS ===")
print("""
Training embeddings requires huge corpora.
Better to use pre-trained:

Popular pre-trained embeddings:
  - Word2Vec Google News (3M words, 300d)
  - GloVe Wikipedia (400K words, 50-300d)
  - FastText (157 languages)

Loading with Gensim:
""")

# Simulated embeddings (normally would load pre-trained)
np.random.seed(42)

# Simulate some word vectors (300 dimensions typical)
embedding_dim = 50

# Create fake but meaningful embeddings
def create_synthetic_embeddings():
    """Create synthetic embeddings for demonstration"""
    # Base vectors for semantic groups
    animal_base = np.random.randn(embedding_dim) * 0.1
    human_base = np.random.randn(embedding_dim) * 0.1
    royalty_base = np.random.randn(embedding_dim) * 0.1
    
    embeddings = {
        # Animals
        'dog': animal_base + np.array([0.1]*embedding_dim),
        'cat': animal_base + np.array([0.12]*embedding_dim),
        'puppy': animal_base + np.array([0.11, 0.05]*25),
        
        # Humans
        'man': human_base + np.array([0.1]*embedding_dim),
        'woman': human_base + np.array([0.08]*embedding_dim),
        'boy': human_base + np.array([0.11, 0.05]*25),
        'girl': human_base + np.array([0.09, 0.05]*25),
        
        # Royalty
        'king': royalty_base + np.array([0.2]*embedding_dim) + human_base,
        'queen': royalty_base + np.array([0.18]*embedding_dim) + human_base,
        'prince': royalty_base + np.array([0.15, 0.1]*25) + human_base,
        'princess': royalty_base + np.array([0.13, 0.1]*25) + human_base,
    }
    return embeddings

embeddings = create_synthetic_embeddings()

print("Sample embedding vectors (first 5 dimensions):")
for word in ['king', 'queen', 'man', 'woman', 'dog']:
    print(f"  {word}: {embeddings[word][:5].round(3)}")
```

## Semantic Similarity

```python
print("\n=== SEMANTIC SIMILARITY ===")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)

print("Cosine similarities:")
pairs = [
    ('king', 'queen'),
    ('king', 'man'),
    ('king', 'dog'),
    ('dog', 'cat'),
    ('man', 'woman'),
]

for w1, w2 in pairs:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"  {w1} - {w2}: {sim:.3f}")

print("""
Similar words have higher cosine similarity!
  - king-queen: Both royalty
  - dog-cat: Both animals
  - king-dog: Very different concepts
""")
```

## Word Analogies

```python
print("\n=== WORD ANALOGIES ===")
print("""
Famous property of word embeddings:

king - man + woman ≈ queen

Vector arithmetic captures relationships!

Other examples:
  - Paris - France + Italy ≈ Rome (capitals)
  - walking - walk + swim ≈ swimming (tense)
  - bigger - big + small ≈ smaller (comparative)
""")

def analogy(a, b, c, embeddings, top_n=3):
    """Find word completing analogy: a is to b as c is to ?"""
    # a - b + c = ?
    target = embeddings[a] - embeddings[b] + embeddings[c]
    
    # Find most similar word
    similarities = []
    for word, vec in embeddings.items():
        if word not in [a, b, c]:
            sim = cosine_similarity(target, vec)
            similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# king - man + woman = ?
result = analogy('king', 'man', 'woman', embeddings)
print("king - man + woman = ?")
for word, sim in result:
    print(f"  {word}: {sim:.3f}")
```

## GloVe: Global Vectors

```python
print("\n=== GloVe ===")
print("""
GloVe (Global Vectors for Word Representation):

Different approach from Word2Vec:
  1. Build word co-occurrence matrix from corpus
  2. Factorize matrix to get embeddings
  3. Combines global statistics with local context

Objective: 
  wᵢ · wⱼ + bᵢ + bⱼ = log(Xᵢⱼ)
  
Where Xᵢⱼ = count of word j appearing near word i

Often performs similarly to Word2Vec.
Popular sizes: 50d, 100d, 200d, 300d
""")
```

## FastText: Subword Embeddings

```python
print("\n=== FASTTEXT ===")
print("""
FastText extends Word2Vec with subword information:

Word = sum of character n-grams

"where" → ["<wh", "whe", "her", "ere", "re>", "<where>"]

Benefits:
  - Handle out-of-vocabulary (OOV) words
  - Better for morphologically rich languages
  - "unhappiness" gets embedding even if never seen

FastText = Word2Vec + character n-grams
""")

def get_ngrams(word, n=3):
    """Get character n-grams for a word"""
    word = f"<{word}>"  # Add boundary markers
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])
    return ngrams

print("Character trigrams for 'where':")
print(f"  {get_ngrams('where', 3)}")

print("\nCharacter trigrams for 'unhappiness':")
print(f"  {get_ngrams('unhappiness', 3)[:8]}...")
```

## Document Embeddings

```python
print("\n=== DOCUMENT EMBEDDINGS ===")
print("""
How to get embedding for entire document?

Simple approaches:
  1. AVERAGE word embeddings
  2. WEIGHTED average (TF-IDF weights)
  3. Sum and normalize

Better approaches:
  - Doc2Vec: Learn document embeddings directly
  - Sentence-BERT: Use transformer models
  
Simple averaging often works well as baseline!
""")

def document_embedding(text, embeddings, method='mean'):
    """Get document embedding by averaging word embeddings"""
    words = text.lower().split()
    word_vecs = [embeddings[w] for w in words if w in embeddings]
    
    if not word_vecs:
        return np.zeros(embedding_dim)
    
    if method == 'mean':
        return np.mean(word_vecs, axis=0)
    elif method == 'sum':
        return np.sum(word_vecs, axis=0)

doc1 = "king queen prince"
doc2 = "dog cat puppy"

emb1 = document_embedding(doc1, embeddings)
emb2 = document_embedding(doc2, embeddings)

print(f"Document similarity:")
print(f"  '{doc1}' vs '{doc2}': {cosine_similarity(emb1, emb2):.3f}")
```

## Key Points

- **Word embeddings**: Dense vector representations of words
- **Distributional hypothesis**: Similar contexts → similar meanings
- **Word2Vec**: Skip-gram and CBOW architectures
- **GloVe**: Based on co-occurrence matrix factorization
- **FastText**: Includes subword information
- **Cosine similarity**: Measure semantic similarity
- **Analogies**: Vector arithmetic captures relationships
- **Document embedding**: Average word vectors

## Reflection Questions

1. Why do dense embeddings capture more semantic information than sparse BoW vectors?
2. How would you handle words not in your pre-trained embedding vocabulary?
3. When might simple averaging of word embeddings fail to represent a document well?
