# Bag of Words and TF-IDF

## Introduction

Before machine learning can work with text, we need to convert words into numbers. Bag of Words and TF-IDF are fundamental techniques for creating numerical text representations.

## The Bag of Words Model

```python
import numpy as np
import pandas as pd
from collections import Counter

print("=== BAG OF WORDS ===")
print("""
BAG OF WORDS (BoW): Represent text as word frequencies

Steps:
  1. Create vocabulary (all unique words)
  2. For each document, count word occurrences
  3. Create vector of word counts

Result: Document → Vector of word counts

Properties:
  - Loses word order (hence "bag")
  - Dimension = vocabulary size
  - Sparse (most entries are 0)
""")

# Sample documents
docs = [
    "I love machine learning",
    "machine learning is great",
    "I love data science",
    "data science is fun"
]

print("Sample documents:")
for i, doc in enumerate(docs):
    print(f"  {i+1}. {doc}")
```

## Building Vocabulary

```python
print("\n=== BUILDING VOCABULARY ===")

def build_vocabulary(documents):
    """Build vocabulary from documents"""
    vocab = set()
    for doc in documents:
        words = doc.lower().split()
        vocab.update(words)
    return sorted(vocab)  # Sort for consistent ordering

vocab = build_vocabulary(docs)
word_to_idx = {word: i for i, word in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
print(f"Words: {vocab}")
print(f"\nWord to index mapping:")
for word, idx in word_to_idx.items():
    print(f"  '{word}': {idx}")
```

## Creating BoW Vectors

```python
print("\n=== BOW VECTORS ===")

def text_to_bow(text, vocab, word_to_idx):
    """Convert text to bag of words vector"""
    vector = np.zeros(len(vocab))
    words = text.lower().split()
    for word in words:
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector

# Create BoW matrix
bow_matrix = np.array([text_to_bow(doc, vocab, word_to_idx) for doc in docs])

print("BoW Matrix:")
df_bow = pd.DataFrame(bow_matrix, columns=vocab, index=[f"Doc{i+1}" for i in range(len(docs))])
print(df_bow.astype(int))

print("\nNote: Each row is a document, each column is a word count")
```

## Using Scikit-Learn

```python
print("\n=== SKLEARN CountVectorizer ===")

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
bow_sklearn = vectorizer.fit_transform(docs)

print(f"Shape: {bow_sklearn.shape}")
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")

print("\nBoW Matrix (sklearn):")
df_sklearn = pd.DataFrame(bow_sklearn.toarray(), 
                          columns=vectorizer.get_feature_names_out(),
                          index=[f"Doc{i+1}" for i in range(len(docs))])
print(df_sklearn)
```

## N-Grams

```python
print("\n=== N-GRAMS ===")
print("""
N-grams: Sequences of N consecutive words

Unigrams (N=1): "machine", "learning", "is"
Bigrams (N=2): "machine learning", "learning is"
Trigrams (N=3): "machine learning is"

Why use N-grams?
  - Capture word order
  - "not good" vs "good" (sentiment)
  - Phrases and expressions
""")

# Bigrams example
vectorizer_bigram = CountVectorizer(ngram_range=(1, 2))
bow_bigram = vectorizer_bigram.fit_transform(docs)

print(f"Features with bigrams: {bow_bigram.shape[1]}")
print(f"\nSample features:")
features = vectorizer_bigram.get_feature_names_out()
for i in range(min(15, len(features))):
    print(f"  {features[i]}")
```

## The Problem with Raw Counts

```python
print("\n=== PROBLEM WITH RAW COUNTS ===")
print("""
Issues with simple word counts:

1. COMMON WORDS dominate:
   "the", "is", "a" appear frequently but carry little meaning

2. DOCUMENT LENGTH bias:
   Longer documents have higher counts

3. RARE but IMPORTANT words:
   Unique words may be most informative

Solution: TF-IDF weighting
""")
```

## TF-IDF: Term Frequency-Inverse Document Frequency

```python
print("\n=== TF-IDF ===")
print("""
TF-IDF weighs words by importance:

TF (Term Frequency): How often word appears in document
  TF = count(word, doc) / total_words(doc)

IDF (Inverse Document Frequency): How rare word is across documents
  IDF = log(N / df) where df = number of docs containing word

TF-IDF = TF × IDF

Effect:
  - High TF-IDF: Word frequent in document, rare overall
  - Low TF-IDF: Word rare in document OR common everywhere
""")

# Manual TF-IDF calculation
def compute_tf(doc):
    """Compute term frequencies"""
    words = doc.lower().split()
    word_counts = Counter(words)
    total = len(words)
    return {word: count/total for word, count in word_counts.items()}

def compute_idf(docs, vocab):
    """Compute inverse document frequencies"""
    N = len(docs)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in docs if word in doc.lower().split())
        idf[word] = np.log(N / (df + 1)) + 1  # Smoothed IDF
    return idf

# Example calculation
tf_doc1 = compute_tf(docs[0])
idf = compute_idf(docs, vocab)

print("TF for Doc 1 ('I love machine learning'):")
for word, tf in tf_doc1.items():
    print(f"  {word}: TF={tf:.3f}, IDF={idf.get(word, 0):.3f}, TF-IDF={tf*idf.get(word, 0):.3f}")
```

## TF-IDF with Scikit-Learn

```python
print("\n=== SKLEARN TfidfVectorizer ===")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

print(f"Shape: {tfidf_matrix.shape}")

print("\nTF-IDF Matrix:")
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(),
                        columns=tfidf_vectorizer.get_feature_names_out(),
                        index=[f"Doc{i+1}" for i in range(len(docs))])
print(df_tfidf.round(3))

print("\nNote: Vectors are L2-normalized by default")
```

## Comparing BoW vs TF-IDF

```python
print("\n=== BOW vs TF-IDF ===")

# Add a document with common words
docs_extended = docs + ["the the the the is is is"]

print("Extended documents:")
for i, doc in enumerate(docs_extended):
    print(f"  {i+1}. {doc}")

# BoW
bow_vec = CountVectorizer()
bow_result = bow_vec.fit_transform(docs_extended)

# TF-IDF  
tfidf_vec = TfidfVectorizer()
tfidf_result = tfidf_vec.fit_transform(docs_extended)

print("\nLast document (repetitive common words):")
print(f"BoW vector sum: {bow_result[-1].sum():.0f}")
print(f"TF-IDF vector sum: {tfidf_result[-1].sum():.3f}")

print("""
TF-IDF gives lower weight to documents with only common words!
""")
```

## Important Parameters

```python
print("\n=== IMPORTANT PARAMETERS ===")
print("""
CountVectorizer / TfidfVectorizer parameters:

max_features: Limit vocabulary size
  - Use top N most frequent words
  - Reduces dimensionality

min_df / max_df: Document frequency thresholds
  - min_df=5: Word must appear in at least 5 docs
  - max_df=0.9: Ignore words in >90% of docs

ngram_range: Include n-grams
  - (1, 1): Only unigrams
  - (1, 2): Unigrams and bigrams

stop_words: Remove stop words
  - 'english': Use built-in list
  - custom list

lowercase: Convert to lowercase (default True)
""")

# Example with parameters
tfidf_custom = TfidfVectorizer(
    max_features=100,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words='english'
)

# With larger corpus
large_docs = [
    "Machine learning algorithms learn from data",
    "Deep learning is a subset of machine learning",
    "Natural language processing handles text data",
    "Data science combines statistics and programming",
    "Neural networks are used in deep learning"
]

tfidf_large = tfidf_custom.fit_transform(large_docs)
print(f"Features: {tfidf_large.shape[1]}")
print(f"Sample features: {tfidf_custom.get_feature_names_out()[:10]}")
```

## Key Points

- **Bag of Words**: Count word occurrences, ignore order
- **Vocabulary**: Set of unique words across all documents
- **N-grams**: Capture word sequences and phrases
- **TF-IDF**: Weight by term frequency × inverse document frequency
- **TF**: How frequent in this document
- **IDF**: How rare across all documents
- **Sparse matrices**: Most entries are zero, use sparse storage

## Reflection Questions

1. Why does BoW lose potentially important information about word order?
2. When would you choose raw word counts over TF-IDF?
3. How do you decide the right vocabulary size for your problem?
