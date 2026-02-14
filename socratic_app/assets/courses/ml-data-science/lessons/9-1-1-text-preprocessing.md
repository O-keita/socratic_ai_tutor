# Text Preprocessing

## Introduction

Text preprocessing transforms raw text into a clean, structured format suitable for machine learning. These foundational steps are essential for any NLP task.

## Why Preprocessing Matters

```python
import numpy as np
import pandas as pd
import re

print("=== TEXT PREPROCESSING ===")
print("""
Raw text is messy:
  - Different cases (Hello, HELLO, hello)
  - Punctuation and special characters
  - Extra whitespace
  - HTML tags, URLs
  - Numbers
  - Stop words (the, is, are)
  - Different word forms (running, ran, runs)

Preprocessing goals:
  1. NORMALIZE text (consistent format)
  2. REDUCE noise (remove irrelevant info)
  3. SIMPLIFY vocabulary (fewer unique tokens)
  4. PREPARE for model input
""")

# Sample text
sample_text = """
<p>Hello World!!! I'm learning NLP with Python 3.11...
Visit https://example.com for MORE info.
The price is $99.99 (or €85).
#MachineLearning #NLP @datascience 
</p>
"""

print("Raw text:")
print(sample_text)
```

## Basic Cleaning

```python
print("\n=== BASIC CLEANING ===")

def basic_clean(text):
    """Basic text cleaning"""
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

cleaned = basic_clean(sample_text)
print("After basic cleaning:")
print(cleaned)
```

## Removing Punctuation and Numbers

```python
print("\n=== PUNCTUATION AND NUMBERS ===")

def remove_punctuation(text):
    """Remove punctuation"""
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    """Remove numbers"""
    return re.sub(r'\d+', '', text)

def keep_alpha_only(text):
    """Keep only alphabetic characters and spaces"""
    return re.sub(r'[^a-zA-Z\s]', '', text)

text = "Hello! This costs $99.99 or €85 in 2024."

print(f"Original: {text}")
print(f"No punctuation: {remove_punctuation(text)}")
print(f"No numbers: {remove_numbers(text)}")
print(f"Alpha only: {keep_alpha_only(text)}")

print("""
Note: Sometimes numbers are important!
  - Product IDs, dates, prices
  - Consider your use case
""")
```

## Tokenization

```python
print("\n=== TOKENIZATION ===")
print("""
TOKENIZATION: Splitting text into individual units (tokens)

Token types:
  - Word tokens: Split on spaces/punctuation
  - Character tokens: Each character is a token
  - Subword tokens: Parts of words (BPE, WordPiece)
  - Sentence tokens: Split into sentences
""")

def simple_tokenize(text):
    """Simple word tokenization"""
    return text.split()

def tokenize_with_regex(text):
    """Tokenize using regex (handles punctuation)"""
    return re.findall(r'\b\w+\b', text.lower())

text = "Hello, I'm learning NLP! It's fascinating."

print(f"Original: {text}")
print(f"Simple split: {simple_tokenize(text)}")
print(f"Regex tokenize: {tokenize_with_regex(text)}")

# Using NLTK (if available)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    print(f"NLTK word_tokenize: {word_tokenize(text)}")
    
    sentences = "Hello world. This is NLP. It's great!"
    print(f"NLTK sent_tokenize: {sent_tokenize(sentences)}")
except ImportError:
    print("(NLTK not installed)")
```

## Stop Words Removal

```python
print("\n=== STOP WORDS ===")
print("""
STOP WORDS: Common words with little meaning
  - Articles: a, an, the
  - Prepositions: in, on, at, for
  - Conjunctions: and, but, or
  - Pronouns: I, you, he, she

Removing them:
  - Reduces vocabulary size
  - Focuses on content words
  
When to KEEP stop words:
  - Sentiment analysis ("not good" vs "good")
  - Phrase detection
  - Some deep learning models
""")

# Common English stop words
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these',
    'of', 'from', 'by', 'with', 'as', 'so', 'if', 'than', 'too', 'very'
}

def remove_stopwords(tokens, stop_words=STOP_WORDS):
    """Remove stop words from token list"""
    return [t for t in tokens if t.lower() not in stop_words]

text = "The quick brown fox jumps over the lazy dog"
tokens = text.lower().split()
filtered = remove_stopwords(tokens)

print(f"Original tokens: {tokens}")
print(f"After removal: {filtered}")
print(f"Removed {len(tokens) - len(filtered)} stop words")
```

## Stemming

```python
print("\n=== STEMMING ===")
print("""
STEMMING: Reducing words to their root form by removing suffixes

running → run
flies → fli (not always real words!)
happily → happili

Algorithms:
  - Porter Stemmer (most common)
  - Snowball Stemmer
  - Lancaster Stemmer (aggressive)

Pros: Simple, fast
Cons: Can produce non-words, over-stems
""")

try:
    from nltk.stem import PorterStemmer, SnowballStemmer
    
    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    
    words = ['running', 'runs', 'ran', 'easily', 'flying', 'flies', 'studies', 'studying']
    
    print(f"{'Word':<12} | {'Porter':<12} | {'Snowball':<12}")
    print("-" * 40)
    for word in words:
        print(f"{word:<12} | {porter.stem(word):<12} | {snowball.stem(word):<12}")
except ImportError:
    # Manual simple stemmer
    def simple_stem(word):
        suffixes = ['ing', 'ed', 'es', 's', 'ly']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    words = ['running', 'runs', 'easily', 'flies']
    print("Simple stemmer:")
    for word in words:
        print(f"  {word} → {simple_stem(word)}")
```

## Lemmatization

```python
print("\n=== LEMMATIZATION ===")
print("""
LEMMATIZATION: Reducing words to dictionary form (lemma)

running → run (verb)
better → good (adjective)
flies → fly (noun or verb)

Uses vocabulary and morphological analysis.

Pros: Produces real words, more accurate
Cons: Slower, needs POS tagging for best results
""")

try:
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    
    words = ['running', 'runs', 'ran', 'better', 'flying', 'flies', 'studies']
    
    print(f"{'Word':<12} | {'As Verb':<12} | {'As Noun':<12}")
    print("-" * 40)
    for word in words:
        verb = lemmatizer.lemmatize(word, pos='v')
        noun = lemmatizer.lemmatize(word, pos='n')
        print(f"{word:<12} | {verb:<12} | {noun:<12}")
except ImportError:
    print("(WordNet not available)")
    print("""
    Example:
      running (verb) → run
      flies (noun) → fly
      flies (verb) → fly
      better (adj) → good
    """)
```

## Complete Preprocessing Pipeline

```python
print("\n=== PREPROCESSING PIPELINE ===")

def preprocess_text(text, remove_stops=True, use_stemming=False):
    """Complete text preprocessing pipeline"""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 5. Tokenize
    tokens = text.split()
    
    # 6. Remove stop words
    if remove_stops:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    
    # 7. Stemming (optional)
    if use_stemming:
        try:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        except:
            pass
    
    return tokens

# Example
sample = """
<p>Machine Learning is transforming how we analyze data!
Visit https://ml-blog.com for the latest updates.
#DataScience #AI</p>
"""

processed = preprocess_text(sample, remove_stops=True, use_stemming=False)
print("Preprocessing result:")
print(f"  Input: {sample[:50]}...")
print(f"  Output: {processed}")
```

## Key Points

- **Lowercasing**: Normalizes case variations
- **Remove noise**: HTML, URLs, special characters
- **Tokenization**: Split text into words/tokens
- **Stop words**: Remove common low-information words
- **Stemming**: Fast, rule-based root extraction
- **Lemmatization**: Slower, dictionary-based, more accurate
- **Pipeline**: Combine steps based on your task

## Reflection Questions

1. When might you want to keep stop words in your text data?
2. What are the trade-offs between stemming and lemmatization?
3. How would preprocessing differ for social media text vs formal documents?
