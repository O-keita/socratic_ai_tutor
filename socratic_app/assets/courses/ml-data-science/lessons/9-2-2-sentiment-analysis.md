# Sentiment Analysis

## Introduction

Sentiment analysis determines the emotional tone of text. It's widely used for analyzing customer reviews, social media, and market research to understand opinions and attitudes.

## What is Sentiment Analysis?

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

print("=== SENTIMENT ANALYSIS ===")
print("""
SENTIMENT ANALYSIS: Determining opinion/emotion in text

Types:
  1. POLARITY: Positive, Negative, Neutral
  2. EMOTION: Joy, Anger, Sadness, Fear, etc.
  3. ASPECT-BASED: Sentiment for specific features
     "The food was great but service was slow"
     → Food: positive, Service: negative

Applications:
  - Customer review analysis
  - Social media monitoring
  - Brand reputation tracking
  - Market research
  - Customer service prioritization
""")
```

## Sample Dataset

```python
print("\n=== SAMPLE DATASET ===")

# Sample reviews with sentiment
reviews = [
    # Positive
    ("This product exceeded my expectations!", "positive"),
    ("Absolutely love it! Best purchase ever.", "positive"),
    ("Amazing quality and fast delivery.", "positive"),
    ("Five stars! Would definitely recommend.", "positive"),
    ("Incredible value for money.", "positive"),
    ("So happy with this purchase!", "positive"),
    ("Works perfectly, exactly as described.", "positive"),
    ("Outstanding customer service!", "positive"),
    
    # Negative
    ("Terrible quality, complete waste of money.", "negative"),
    ("Very disappointed, not as advertised.", "negative"),
    ("Would not recommend to anyone.", "negative"),
    ("Broke after one week, terrible product.", "negative"),
    ("Worst purchase I've ever made.", "negative"),
    ("Horrible experience, want my money back.", "negative"),
    ("Poor quality and slow shipping.", "negative"),
    ("Total garbage, avoid at all costs.", "negative"),
    
    # Neutral
    ("The product is okay, nothing special.", "neutral"),
    ("It works as expected, average quality.", "neutral"),
    ("Decent for the price, could be better.", "neutral"),
    ("Not bad, not great, just okay.", "neutral"),
    ("Meets basic expectations.", "neutral"),
    ("Average product, nothing to complain about.", "neutral"),
]

df = pd.DataFrame(reviews, columns=['text', 'sentiment'])

print("Dataset sample:")
print(df.head(10))
print(f"\nSentiment distribution:")
print(df['sentiment'].value_counts())
```

## Preprocessing for Sentiment

```python
print("\n=== PREPROCESSING ===")

def preprocess_for_sentiment(text):
    """Preprocess text while preserving sentiment cues"""
    # Lowercase
    text = text.lower()
    
    # Expand contractions
    contractions = {
        "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "won't": "will not",
        "can't": "cannot", "couldn't": "could not",
        "wouldn't": "would not", "shouldn't": "should not",
        "isn't": "is not", "aren't": "are not",
        "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Keep exclamation and question marks (emotional cues)
    text = re.sub(r'[^\w\s!?]', '', text)
    
    # Handle repeated characters (loooove -> love)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    return text

print("Preprocessing examples:")
examples = ["I don't like this!!!", "This is soooo good!", "Wouldn't recommend."]
for ex in examples:
    print(f"  '{ex}' → '{preprocess_for_sentiment(ex)}'")
```

## Building Sentiment Classifier

```python
print("\n=== TRAINING CLASSIFIER ===")

# Preprocess
df['text_clean'] = df['text'].apply(preprocess_for_sentiment)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['sentiment'], 
    test_size=0.3, random_state=42, stratify=df['sentiment']
)

# Vectorize
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=500
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

## Analyzing Sentiment Words

```python
print("\n=== SENTIMENT WORDS ===")

def get_sentiment_words(model, vectorizer, class_name, n=8):
    """Get words most associated with sentiment"""
    feature_names = vectorizer.get_feature_names_out()
    class_idx = list(model.classes_).index(class_name)
    
    coef = model.coef_[class_idx]
    top_idx = coef.argsort()[-n:][::-1]
    bottom_idx = coef.argsort()[:n]
    
    positive_words = [(feature_names[i], coef[i]) for i in top_idx]
    negative_words = [(feature_names[i], coef[i]) for i in bottom_idx]
    
    return positive_words, negative_words

print("Words most associated with POSITIVE sentiment:")
pos_words, _ = get_sentiment_words(model, vectorizer, 'positive')
for word, score in pos_words:
    print(f"  {word}: {score:.3f}")

print("\nWords most associated with NEGATIVE sentiment:")
neg_words, _ = get_sentiment_words(model, vectorizer, 'negative')
for word, score in neg_words:
    print(f"  {word}: {score:.3f}")
```

## Handling Negation

```python
print("\n=== HANDLING NEGATION ===")
print("""
Negation flips sentiment:
  "good" → positive
  "not good" → negative!

Approaches:
  1. N-grams: Capture "not good" as single feature
  2. Negation marking: Add NOT_ prefix to words after negation
  3. Dependency parsing: Understand grammatical structure
""")

def mark_negation(text):
    """Mark words following negation"""
    negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing'}
    words = text.split()
    result = []
    negate = False
    
    for word in words:
        if word.lower() in negation_words:
            negate = True
            result.append(word)
        elif negate:
            result.append(f"NOT_{word}")
            if word.endswith(('.', '!', '?', ',')):
                negate = False
        else:
            result.append(word)
    
    return ' '.join(result)

examples = ["I do not like this product", "never buy this again", "not good at all"]
print("Negation marking examples:")
for ex in examples:
    print(f"  '{ex}' → '{mark_negation(ex)}'")
```

## Predicting Sentiment

```python
print("\n=== PREDICTING SENTIMENT ===")

def predict_sentiment(texts, model, vectorizer):
    """Predict sentiment for new texts"""
    # Preprocess
    texts_clean = [preprocess_for_sentiment(t) for t in texts]
    
    # Vectorize
    X = vectorizer.transform(texts_clean)
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities

# New texts
new_reviews = [
    "This is absolutely fantastic! Love it!",
    "Terrible product, completely useless.",
    "It's okay, nothing special but works.",
    "Not bad, pretty good actually.",
    "Would not recommend, very disappointed."
]

predictions, probabilities = predict_sentiment(new_reviews, model, vectorizer)

print("Sentiment predictions:")
for text, pred, prob in zip(new_reviews, predictions, probabilities):
    print(f"\n  Text: '{text[:40]}...'")
    print(f"  Sentiment: {pred}")
    conf = max(prob)
    print(f"  Confidence: {conf:.3f}")
```

## Sentiment Scoring

```python
print("\n=== SENTIMENT SCORING ===")
print("""
Instead of categories, sometimes want a score:
  - Negative: -1 to 0
  - Positive: 0 to +1
  
Or intensity score:
  "okay" → 0.2
  "good" → 0.5
  "great" → 0.7
  "amazing" → 0.9
""")

def sentiment_score(text, model, vectorizer):
    """Get sentiment score (-1 to 1)"""
    text_clean = preprocess_for_sentiment(text)
    X = vectorizer.transform([text_clean])
    
    probs = model.predict_proba(X)[0]
    
    # Weighted score: pos contributes +, neg contributes -, neutral is 0
    class_weights = {'positive': 1, 'neutral': 0, 'negative': -1}
    
    score = 0
    for cls, prob in zip(model.classes_, probs):
        score += class_weights[cls] * prob
    
    return score

print("Sentiment scores:")
for text in new_reviews:
    score = sentiment_score(text, model, vectorizer)
    print(f"  Score: {score:+.3f} | '{text[:35]}...'")
```

## Aspect-Based Sentiment

```python
print("\n=== ASPECT-BASED SENTIMENT ===")
print("""
Analyze sentiment for specific ASPECTS:

"The food was delicious but the service was slow and the price was reasonable."

Aspects:
  - food: positive
  - service: negative
  - price: positive

Approaches:
  1. Define aspects + keywords
  2. Extract sentences mentioning each aspect
  3. Analyze sentiment for each aspect
""")

def simple_aspect_sentiment(text, aspects):
    """Simple aspect-based sentiment (keyword matching)"""
    sentiment_words = {
        'positive': ['good', 'great', 'excellent', 'amazing', 'love', 'delicious', 'fast', 'helpful'],
        'negative': ['bad', 'poor', 'terrible', 'slow', 'rude', 'expensive', 'awful', 'worst']
    }
    
    text_lower = text.lower()
    results = {}
    
    for aspect, keywords in aspects.items():
        # Check if aspect is mentioned
        if any(kw in text_lower for kw in keywords):
            # Simple sentiment detection around aspect
            pos_count = sum(w in text_lower for w in sentiment_words['positive'])
            neg_count = sum(w in text_lower for w in sentiment_words['negative'])
            
            if pos_count > neg_count:
                results[aspect] = 'positive'
            elif neg_count > pos_count:
                results[aspect] = 'negative'
            else:
                results[aspect] = 'neutral'
    
    return results

aspects = {
    'food': ['food', 'meal', 'dish', 'taste'],
    'service': ['service', 'staff', 'waiter', 'waitress'],
    'price': ['price', 'cost', 'value', 'expensive', 'cheap']
}

review = "The food was delicious but the service was slow and the price was reasonable."
result = simple_aspect_sentiment(review, aspects)

print(f"Review: '{review}'")
print(f"Aspect sentiments: {result}")
```

## Key Points

- **Sentiment analysis**: Determine opinion/emotion in text
- **Polarity**: Positive, negative, neutral classification
- **Negation handling**: Critical for accuracy ("not good" is negative)
- **Preprocessing**: Preserve emotional cues (!, ?)
- **Feature importance**: Identify sentiment-bearing words
- **Sentiment scoring**: Continuous score instead of categories
- **Aspect-based**: Sentiment for specific product/service features

## Reflection Questions

1. How would you handle sarcasm in sentiment analysis?
2. Why is negation handling particularly important for sentiment analysis?
3. What are the limitations of lexicon-based vs machine learning approaches to sentiment?
