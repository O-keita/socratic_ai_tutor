# Topic Modeling

## Introduction

Topic modeling discovers abstract topics within a collection of documents. It's an unsupervised technique that reveals the hidden thematic structure in large text corpora.

## What is Topic Modeling?

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

print("=== TOPIC MODELING ===")
print("""
TOPIC MODELING: Discover hidden topics in documents

Intuition:
  - Documents are mixtures of topics
  - Topics are distributions over words
  
Example:
  Document about "ML in healthcare" might be:
    40% Machine Learning topic
    50% Healthcare topic  
    10% Business topic

Topics are defined by word probabilities:
  ML topic: "model", "training", "data", "algorithm"
  Healthcare topic: "patient", "doctor", "treatment", "hospital"

Applications:
  - Document organization
  - Content recommendation
  - Trend analysis
  - Research literature analysis
""")
```

## Sample Document Collection

```python
print("\n=== SAMPLE DOCUMENTS ===")

documents = [
    # Tech/AI
    "Machine learning algorithms can analyze large datasets efficiently.",
    "Deep neural networks have revolutionized artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Cloud computing enables scalable machine learning solutions.",
    
    # Sports
    "The football team won the championship game last night.",
    "Basketball players train rigorously for the season.",
    "The tennis match attracted thousands of spectators.",
    "Running marathons requires months of dedicated training.",
    
    # Finance
    "Stock market investors analyze quarterly earnings reports.",
    "The central bank raised interest rates to control inflation.",
    "Cryptocurrency trading has become increasingly popular.",
    "Investment portfolios should be diversified for risk management.",
    
    # Healthcare
    "Doctors recommend regular exercise for heart health.",
    "Medical researchers are developing new cancer treatments.",
    "Hospitals are adopting electronic health record systems.",
    "Preventive care can reduce healthcare costs significantly."
]

print(f"Number of documents: {len(documents)}")
print("\nSample documents:")
for i, doc in enumerate(documents[:4]):
    print(f"  {i+1}. {doc[:50]}...")
```

## Latent Dirichlet Allocation (LDA)

```python
print("\n=== LDA TOPIC MODEL ===")
print("""
LDA (Latent Dirichlet Allocation):

Generative model assumptions:
  1. Each DOCUMENT is a mixture of topics
  2. Each TOPIC is a mixture of words
  
Process (generative story):
  - For each document:
    - Choose topic distribution (Dirichlet)
    - For each word:
      - Choose a topic from distribution
      - Choose a word from that topic's word distribution

Training finds topics that best explain the documents.
""")

# Prepare documents
vectorizer = CountVectorizer(
    max_features=1000,
    stop_words='english',
    max_df=0.95,
    min_df=1
)

doc_term_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Document-term matrix shape: {doc_term_matrix.shape}")
```

## Training LDA

```python
print("\n=== TRAINING LDA ===")

n_topics = 4

lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='online',
    max_iter=20
)

# Fit model
doc_topics = lda.fit_transform(doc_term_matrix)

print(f"Number of topics: {n_topics}")
print(f"Document-topic matrix shape: {doc_topics.shape}")

# Display topics
def display_topics(model, feature_names, n_words=8):
    """Display top words for each topic"""
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics.append(top_words)
        print(f"\nTopic {topic_idx + 1}:")
        print(f"  {', '.join(top_words)}")
    return topics

print("\n=== DISCOVERED TOPICS ===")
topics = display_topics(lda, feature_names)
```

## Document-Topic Distribution

```python
print("\n=== DOCUMENT-TOPIC DISTRIBUTIONS ===")

print("Topic distribution for each document:\n")
print(f"{'Document':<50} | Topic Weights")
print("-" * 80)

for i, (doc, topic_dist) in enumerate(zip(documents[:8], doc_topics[:8])):
    topic_str = " ".join([f"T{j+1}:{w:.2f}" for j, w in enumerate(topic_dist)])
    print(f"{doc[:47]+'...':<50} | {topic_str}")

# Dominant topic per document
dominant_topics = doc_topics.argmax(axis=1)
print("\n\nDominant topic per document:")
for i, (doc, topic) in enumerate(zip(documents[:8], dominant_topics[:8])):
    print(f"  Doc {i+1}: Topic {topic + 1}")
```

## Non-Negative Matrix Factorization (NMF)

```python
print("\n=== NMF TOPIC MODEL ===")
print("""
NMF (Non-negative Matrix Factorization):

Alternative to LDA:
  - Factorize document-term matrix: V ≈ W × H
  - W: Document-topic matrix
  - H: Topic-term matrix
  - All values non-negative

Differences from LDA:
  - No probabilistic interpretation
  - Often produces more coherent topics
  - Faster on large datasets
  - Works better with TF-IDF features
""")

# Use TF-IDF for NMF
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    max_df=0.95,
    min_df=1
)

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Train NMF
nmf = NMF(n_components=4, random_state=42, max_iter=500)
doc_topics_nmf = nmf.fit_transform(tfidf_matrix)

print("\n=== NMF TOPICS ===")
display_topics(nmf, tfidf_feature_names)
```

## Choosing Number of Topics

```python
print("\n=== CHOOSING NUMBER OF TOPICS ===")
print("""
Methods for selecting K (number of topics):

1. PERPLEXITY (LDA):
   - Lower is better
   - Measures how well model predicts held-out data
   - Can overfit to training data

2. COHERENCE SCORE:
   - Measures topic quality
   - Higher is better
   - Based on word co-occurrence patterns

3. DOMAIN KNOWLEDGE:
   - What makes sense for your application?
   - Human interpretability matters

4. ELBOW METHOD:
   - Plot metric vs K
   - Find "elbow" where improvement slows
""")

# Perplexity for different K values
k_values = [2, 3, 4, 5, 6]
perplexities = []

for k in k_values:
    lda_k = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=20)
    lda_k.fit(doc_term_matrix)
    perplexity = lda_k.perplexity(doc_term_matrix)
    perplexities.append(perplexity)
    print(f"  K={k}: Perplexity = {perplexity:.2f}")

print("\nNote: Perplexity should decrease but watch for overfitting")
```

## Topic Coherence

```python
print("\n=== TOPIC COHERENCE ===")
print("""
Topic coherence measures how interpretable topics are:

C_v coherence:
  - Based on word co-occurrence
  - Higher values = more coherent topics
  - Range: 0 to 1 typically

Manual coherence (simplified):
  - Count how often top words appear together
  - More co-occurrence = more coherent
""")

def simple_coherence(topic_words, documents, top_n=5):
    """Simple coherence: pairwise word co-occurrence"""
    words = topic_words[:top_n]
    cooccurrence = 0
    pairs = 0
    
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            pairs += 1
            for doc in documents:
                if w1 in doc.lower() and w2 in doc.lower():
                    cooccurrence += 1
                    break
    
    return cooccurrence / pairs if pairs > 0 else 0

print("Simple coherence for each topic:")
for i, topic in enumerate(topics):
    coh = simple_coherence(topic, documents)
    print(f"  Topic {i+1}: {coh:.3f}")
```

## Visualizing Topics

```python
print("\n=== VISUALIZING TOPICS ===")
print("""
Common visualizations:

1. WORD CLOUDS: Size = word importance in topic
2. BAR CHARTS: Top N words per topic with weights
3. TOPIC HEATMAP: Document-topic distribution
4. pyLDAvis: Interactive visualization
   - Topic distances
   - Word relevance
   - Topic sizes
""")

# Simple bar chart data
print("\nTopic word weights (for plotting):")
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[:-6:-1]
    top_words = [(feature_names[i], topic[i]) for i in top_indices]
    print(f"\nTopic {topic_idx + 1}:")
    for word, weight in top_words:
        bar = '█' * int(weight * 10)
        print(f"  {word:<15} {bar} ({weight:.2f})")
```

## Applying to New Documents

```python
print("\n=== INFERENCE ON NEW DOCUMENTS ===")

new_documents = [
    "The neural network achieved high accuracy on image classification.",
    "The team scored three goals in the second half.",
    "Interest rates affect mortgage payments significantly.",
]

# Transform new documents
new_doc_term = vectorizer.transform(new_documents)
new_topic_dist = lda.transform(new_doc_term)

print("Topic distributions for new documents:")
for doc, dist in zip(new_documents, new_topic_dist):
    dominant = dist.argmax() + 1
    confidence = dist.max()
    print(f"\n  '{doc[:45]}...'")
    print(f"  Dominant topic: {dominant} (confidence: {confidence:.2f})")
    print(f"  Distribution: {dist.round(2)}")
```

## Key Points

- **Topic modeling**: Unsupervised discovery of themes in text
- **LDA**: Probabilistic model, documents as topic mixtures
- **NMF**: Matrix factorization approach, often faster
- **K selection**: Perplexity, coherence, domain knowledge
- **Coherence**: Measures topic interpretability
- **Applications**: Document organization, recommendations, trend analysis

## Reflection Questions

1. How would you validate that discovered topics are meaningful?
2. When would you choose NMF over LDA or vice versa?
3. How might topics change as you add more documents to your corpus?
