# Logistic Regression

## Introduction

Logistic regression is the fundamental algorithm for binary classification. Despite its name, it's a classification algorithm that models the probability of class membership using a logistic (sigmoid) function.

## From Linear to Logistic

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)

print("=== LOGISTIC REGRESSION ===")
print("""
Problem with linear regression for classification:
  - Outputs can be < 0 or > 1 (not valid probabilities)
  - Doesn't bound predictions

Solution: Use logistic (sigmoid) function

Sigmoid: σ(z) = 1 / (1 + e^(-z))
  - Always between 0 and 1
  - Maps any real number to probability

Model:
  z = β₀ + β₁x₁ + ... + βₙxₙ  (linear combination)
  P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
""")
```

## Binary Classification Example

```python
print("\n=== BINARY CLASSIFICATION ===")

# Generate sample data
n = 200
study_hours = np.random.uniform(1, 10, n)
pass_exam = (study_hours + np.random.randn(n) * 1.5 > 5).astype(int)

X = study_hours.reshape(-1, 1)
y = pass_exam

print(f"Data: {n} students")
print(f"Pass rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"\nModel coefficients:")
print(f"  Intercept (β₀): {model.intercept_[0]:.3f}")
print(f"  Study hours coefficient (β₁): {model.coef_[0][0]:.3f}")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(f"\nSample predictions (first 5):")
for i in range(5):
    print(f"  Hours={X_test[i,0]:.1f}: P(pass)={y_prob[i,1]:.3f}, Predicted={y_pred[i]}, Actual={y_test[i]}")
```

## The Sigmoid Function

```python
print("\n=== SIGMOID FUNCTION ===")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("Sigmoid values at key points:")
z_values = [-6, -4, -2, -1, 0, 1, 2, 4, 6]
for z in z_values:
    print(f"  z={z:+2d}: σ(z)={sigmoid(z):.4f}")

print("""
Key properties:
  - σ(0) = 0.5 (decision boundary)
  - σ(z) → 1 as z → +∞
  - σ(z) → 0 as z → -∞
  - Symmetric: σ(-z) = 1 - σ(z)
""")
```

## Interpreting Coefficients

```python
print("\n=== INTERPRETING COEFFICIENTS ===")
print("""
In logistic regression, coefficients are in LOG-ODDS:

Log-odds (logit):
  log(P/(1-P)) = β₀ + β₁x₁ + ... + βₙxₙ

Interpretation of β₁:
  - 1-unit increase in x₁ increases log-odds by β₁
  - Multiply odds by exp(β₁)

Odds Ratio:
  OR = exp(β₁)
  - OR > 1: Positive relationship
  - OR < 1: Negative relationship
  - OR = 1: No relationship
""")

# Calculate odds ratio
beta_1 = model.coef_[0][0]
odds_ratio = np.exp(beta_1)

print(f"Coefficient for study hours: {beta_1:.3f}")
print(f"Odds ratio: {odds_ratio:.3f}")
print(f"Interpretation: Each additional hour of study multiplies odds of passing by {odds_ratio:.2f}")

# Example
print("\nExample calculation:")
print(f"  If P(pass|4hrs) = 0.3, odds = 0.3/0.7 = {0.3/0.7:.3f}")
print(f"  At 5 hours: odds = {0.3/0.7:.3f} × {odds_ratio:.2f} = {0.3/0.7 * odds_ratio:.3f}")
print(f"  Converting back: P(pass|5hrs) = odds/(1+odds) = {0.3/0.7 * odds_ratio / (1 + 0.3/0.7 * odds_ratio):.3f}")
```

## Model Evaluation

```python
print("\n=== EVALUATION METRICS ===")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
print("""
           Predicted
            0     1
Actual  0 [TN    FP]
        1 [FN    TP]
""")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

print("""
Metrics explained:
  - Precision: TP / (TP + FP) - "Of predicted positive, how many correct?"
  - Recall: TP / (TP + FN) - "Of actual positive, how many found?"
  - F1: Harmonic mean of precision and recall
""")
```

## Decision Threshold

```python
print("\n=== DECISION THRESHOLD ===")
print("""
Default: Predict class 1 if P(y=1) > 0.5

Can adjust threshold:
  - Lower threshold: More predictions of class 1
    → Higher recall, lower precision
  - Higher threshold: Fewer predictions of class 1
    → Lower recall, higher precision

Use cases:
  - Medical screening: Low threshold (don't miss disease)
  - Spam detection: High threshold (don't block good email)
""")

thresholds = [0.3, 0.5, 0.7]
for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    pred_positive = y_pred_thresh.sum()
    print(f"Threshold {thresh}: Accuracy={acc:.3f}, Predicted positive={pred_positive}")
```

## Multiclass Logistic Regression

```python
print("\n=== MULTICLASS CLASSIFICATION ===")
print("""
For more than 2 classes:

1. ONE-VS-REST (OvR):
   - Train K binary classifiers
   - Each: class i vs all other classes
   - Predict class with highest probability

2. MULTINOMIAL (Softmax):
   - Single model for all classes
   - Uses softmax function
   - P(y=k) = exp(zₖ) / Σexp(zⱼ)
""")

from sklearn.datasets import load_iris

# Load iris dataset (3 classes)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Multinomial logistic regression
model_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
model_multi.fit(X_train_i, y_train_i)

print(f"Iris dataset: {len(iris.target_names)} classes")
print(f"Accuracy: {model_multi.score(X_test_i, y_test_i):.3f}")

# Probabilities
probs = model_multi.predict_proba(X_test_i[:3])
print("\nSample probability predictions:")
for i, (prob, actual) in enumerate(zip(probs, y_test_i[:3])):
    print(f"  Sample {i}: {prob.round(3)}, Predicted: {prob.argmax()}, Actual: {actual}")
```

## Regularization in Logistic Regression

```python
print("\n=== REGULARIZATION ===")
print("""
Logistic regression in sklearn uses regularization by default!

Parameters:
  - penalty: 'l1', 'l2', 'elasticnet', 'none'
  - C: Inverse of regularization strength (smaller = more regularization)
    - C = 1/λ
    - Default C=1.0
    
Solvers:
  - 'lbfgs': L2 only (default)
  - 'liblinear': L1 or L2
  - 'saga': L1, L2, or ElasticNet
""")

# Compare different regularization
Cs = [0.01, 0.1, 1, 10, 100]
print("\nEffect of C (inverse regularization):")
for C in Cs:
    model_reg = LogisticRegression(C=C, max_iter=1000)
    model_reg.fit(X_train_i, y_train_i)
    acc = model_reg.score(X_test_i, y_test_i)
    coef_norm = np.linalg.norm(model_reg.coef_)
    print(f"  C={C:>5}: Accuracy={acc:.3f}, Coef norm={coef_norm:.3f}")
```

## Key Points

- **Logistic regression**: Classification using sigmoid function
- **Outputs probabilities**: P(y=1|x) between 0 and 1
- **Coefficients**: Interpret as log-odds (use exp() for odds ratio)
- **Threshold**: Default 0.5, adjust for precision-recall tradeoff
- **Multiclass**: One-vs-Rest or Multinomial (softmax)
- **Regularization**: Use C parameter (inverse of λ)
- **Not just for binary**: Works for multiclass via softmax

## Reflection Questions

1. Why can't we use linear regression for classification?
2. How would you choose a decision threshold for a medical diagnosis system?
3. What does an odds ratio of 2.5 mean in practical terms?
