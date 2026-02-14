# Text Classification

## Introduction

Text classification assigns predefined categories to text documents. It's one of the most common NLP tasks with applications from spam detection to sentiment analysis.

## Text Classification Overview

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("=== TEXT CLASSIFICATION ===")
print("""
TASK: Assign categories to text documents

Examples:
  - Spam detection (spam vs ham)
  - Sentiment analysis (positive/negative/neutral)
  - Topic classification (sports, politics, tech)
  - Intent detection (question, statement, command)
  - Language identification

Pipeline:
  1. Collect labeled text data
  2. Preprocess text
  3. Convert to numerical features (TF-IDF, embeddings)
  4. Train classifier
  5. Evaluate and deploy
""")
```

## Sample Dataset

```python
print("\n=== SAMPLE DATASET ===")

# Create sample dataset
texts = [
    "I love this product! It works perfectly.",
    "Great quality and fast shipping.",
    "Absolutely amazing experience!",
    "Best purchase I've ever made.",
    "Highly recommend to everyone.",
    "Terrible product, waste of money.",
    "Very disappointed with the quality.",
    "Would not recommend to anyone.",
    "Complete garbage, want refund.",
    "Worst experience ever, very bad.",
    "The product is okay, nothing special.",
    "Average quality for the price.",
    "It works but could be better.",
    "Decent product, met expectations.",
    "Not bad, not great, just okay."
]

labels = [
    'positive', 'positive', 'positive', 'positive', 'positive',
    'negative', 'negative', 'negative', 'negative', 'negative',
    'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
]

df = pd.DataFrame({'text': texts, 'label': labels})

print("Sample data:")
print(df.head(8))
print(f"\nClass distribution:")
print(df['label'].value_counts())
```

## Feature Extraction

```python
print("\n=== FEATURE EXTRACTION ===")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nFeature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"\nSample features: {list(vectorizer.vocabulary_.keys())[:10]}")
```

## Naive Bayes Classifier

```python
print("\n=== NAIVE BAYES ===")
print("""
Multinomial Naive Bayes is popular for text:
  - Fast to train
  - Works well with sparse data
  - Good baseline for text classification
  
P(class|document) ∝ P(class) × ∏P(word|class)

Assumes word independence (naive assumption).
""")

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes Results:")
print(classification_report(y_test, y_pred_nb))
```

## Logistic Regression

```python
print("\n=== LOGISTIC REGRESSION ===")
print("""
Often performs better than Naive Bayes:
  - Linear model, fast training
  - Regularization prevents overfitting
  - Interpretable coefficients
""")

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)

print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))

# Feature importance
def get_top_features(model, vectorizer, class_names, n=5):
    """Get most important features per class"""
    feature_names = vectorizer.get_feature_names_out()
    for i, class_name in enumerate(class_names):
        if hasattr(model, 'coef_'):
            top_idx = model.coef_[i].argsort()[-n:][::-1]
            top_features = [feature_names[j] for j in top_idx]
            print(f"  {class_name}: {top_features}")

print("\nTop features per class:")
get_top_features(lr_model, vectorizer, lr_model.classes_)
```

## Support Vector Machine

```python
print("\n=== SVM FOR TEXT ===")
print("""
Linear SVM often best for high-dimensional text data:
  - Effective in high dimensions
  - Works well with sparse features
  - Maximum margin principle
""")

from sklearn.svm import LinearSVC

svm_model = LinearSVC(random_state=42, max_iter=1000)
svm_model.fit(X_train_tfidf, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf)

print("SVM Results:")
print(classification_report(y_test, y_pred_svm))
```

## Model Comparison

```python
print("\n=== MODEL COMPARISON ===")

from sklearn.ensemble import RandomForestClassifier

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"{'Model':<25} | {'Accuracy':>10}")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<25} | {acc:>10.3f}")
```

## Cross-Validation

```python
print("\n=== CROSS-VALIDATION ===")

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Create pipeline (vectorizer + classifier)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# Cross-validation on all data
scores = cross_val_score(pipeline, df['text'], df['label'], cv=3, scoring='accuracy')

print("Logistic Regression CV Scores:")
print(f"  Scores: {scores.round(3)}")
print(f"  Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Hyperparameter Tuning

```python
print("\n=== HYPERPARAMETER TUNING ===")

from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'tfidf__max_features': [500, 1000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(df['text'], df['label'])

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

## Making Predictions

```python
print("\n=== MAKING PREDICTIONS ===")

# Train final model
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
final_pipeline.fit(df['text'], df['label'])

# New texts
new_texts = [
    "This is wonderful! I'm so happy with it.",
    "Total waste of money, very unhappy.",
    "It does what it's supposed to do."
]

predictions = final_pipeline.predict(new_texts)
probabilities = final_pipeline.predict_proba(new_texts)

print("Predictions on new text:")
for text, pred, prob in zip(new_texts, predictions, probabilities):
    print(f"\n  Text: '{text[:40]}...'")
    print(f"  Prediction: {pred}")
    print(f"  Probabilities: {dict(zip(final_pipeline.classes_, prob.round(3)))}")
```

## Multi-Label Classification

```python
print("\n=== MULTI-LABEL CLASSIFICATION ===")
print("""
Sometimes documents belong to MULTIPLE categories:

Single-label: Document → One class
Multi-label: Document → Multiple classes (0 or more)

Approaches:
  1. Binary Relevance: Separate classifier per label
  2. Classifier Chains: Sequential classifiers
  3. Multi-label specific algorithms

sklearn: MultiOutputClassifier, ClassifierChain
""")

from sklearn.multiclass import OneVsRestClassifier

# Example: Topics that can co-occur
print("Example: Article about 'tech business news'")
print("  Could belong to: [tech, business, news]")
print("  Multi-label approach needed!")
```

## Key Points

- **Text classification**: Assign categories to documents
- **Feature extraction**: TF-IDF most common, embeddings for deep learning
- **Common classifiers**: Naive Bayes, Logistic Regression, SVM
- **Pipeline**: Combine preprocessing and classification
- **Cross-validation**: Essential for reliable evaluation
- **Hyperparameter tuning**: GridSearchCV for best parameters
- **Multi-label**: When documents can have multiple categories

## Reflection Questions

1. Why is Naive Bayes often a good baseline for text classification despite its "naive" independence assumption?
2. How would you handle a highly imbalanced text classification problem?
3. When might word embeddings outperform TF-IDF for text classification?
