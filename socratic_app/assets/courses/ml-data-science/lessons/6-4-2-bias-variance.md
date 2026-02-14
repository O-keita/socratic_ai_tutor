# Bias-Variance Tradeoff

## Introduction

The bias-variance tradeoff is one of the most fundamental concepts in machine learning. Understanding it helps you diagnose model problems and choose appropriate complexity.

## Understanding Bias and Variance

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

print("=== BIAS-VARIANCE TRADEOFF ===")
print("""
EXPECTED ERROR = BIAS² + VARIANCE + IRREDUCIBLE ERROR

BIAS:
  - Error from wrong assumptions in the model
  - "How far is the average prediction from truth?"
  - High bias → Underfitting
  - Examples: Linear model for non-linear data

VARIANCE:
  - Error from sensitivity to training data fluctuations
  - "How much do predictions vary across different training sets?"
  - High variance → Overfitting
  - Examples: Very deep trees, high-degree polynomials

IRREDUCIBLE ERROR:
  - Noise inherent in the data
  - Cannot be reduced by any model
  - Sets the floor for achievable error
""")
```

## Visualizing the Tradeoff

```python
print("\n=== DEMONSTRATING BIAS-VARIANCE ===")

# True function
def true_function(x):
    return np.sin(2 * x) + 0.5 * x

# Generate multiple training sets
n_datasets = 50
n_samples = 30
n_test = 100

x_test = np.linspace(0, 4, n_test)
y_true = true_function(x_test)

def train_and_predict(model_class, degree=None, max_depth=None):
    """Train model on multiple datasets, return predictions"""
    all_predictions = []
    
    for seed in range(n_datasets):
        np.random.seed(seed)
        x_train = np.random.uniform(0, 4, n_samples)
        y_train = true_function(x_train) + np.random.normal(0, 0.3, n_samples)
        
        if degree is not None:  # Polynomial
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('reg', LinearRegression())
            ])
            model.fit(x_train.reshape(-1, 1), y_train)
            pred = model.predict(x_test.reshape(-1, 1))
        else:  # Decision Tree
            model = DecisionTreeRegressor(max_depth=max_depth)
            model.fit(x_train.reshape(-1, 1), y_train)
            pred = model.predict(x_test.reshape(-1, 1))
        
        all_predictions.append(pred)
    
    return np.array(all_predictions)

# Compare models
print("Polynomial Regression with different degrees:")
print(f"{'Degree':>8} | {'Bias²':>10} | {'Variance':>10} | {'Total Error':>12}")
print("-" * 50)

for degree in [1, 3, 5, 10, 15]:
    preds = train_and_predict(None, degree=degree)
    
    # Bias: (mean prediction - true)²
    mean_pred = preds.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_true) ** 2)
    
    # Variance: average variance of predictions
    variance = np.mean(preds.var(axis=0))
    
    total_error = bias_sq + variance
    
    print(f"{degree:>8} | {bias_sq:>10.4f} | {variance:>10.4f} | {total_error:>12.4f}")

print("\nNote: Degree 1 has high bias, Degree 15 has high variance")
```

## Decision Trees and Depth

```python
print("\n=== DECISION TREES: DEPTH vs BIAS-VARIANCE ===")

print(f"{'Max Depth':>10} | {'Bias²':>10} | {'Variance':>10} | {'Total Error':>12}")
print("-" * 50)

for max_depth in [1, 2, 5, 10, None]:
    preds = train_and_predict(None, max_depth=max_depth)
    
    mean_pred = preds.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(preds.var(axis=0))
    total_error = bias_sq + variance
    
    depth_str = str(max_depth) if max_depth else "None"
    print(f"{depth_str:>10} | {bias_sq:>10.4f} | {variance:>10.4f} | {total_error:>12.4f}")

print("""
Key observations:
  - Shallow trees: High bias (underfitting)
  - Deep trees: High variance (overfitting)
  - Optimal depth: Minimizes total error
""")
```

## Model Complexity Spectrum

```python
print("\n=== MODEL COMPLEXITY SPECTRUM ===")
print("""
                LOW COMPLEXITY ◄──────────────► HIGH COMPLEXITY
                
Model:          Linear Reg     Ridge/Lasso    Polynomial    Deep Trees
                K-large        SVM(small C)   RF           Neural Nets
                               
Bias:           HIGH ◄─────────────────────────────────────► LOW
                "Too simple"                               "Can fit anything"
                
Variance:       LOW ◄──────────────────────────────────────► HIGH
                "Stable"                                   "Sensitive to data"
                
Error Pattern:  Underfitting ◄─────────────────────────────► Overfitting
                Train ≈ Test (both bad)                    Train << Test
                
What to do:     Add features    Find optimal    Regularize
                More complex    complexity      Simplify
                model                           More data
""")
```

## Diagnosing Problems

```python
print("\n=== DIAGNOSING UNDERFITTING vs OVERFITTING ===")
print("""
UNDERFITTING (High Bias):
  Symptoms:
    - High training error
    - High test error
    - Training ≈ Test error
    
  Solutions:
    - Use more complex model
    - Add more features
    - Reduce regularization
    - Train longer (neural nets)

OVERFITTING (High Variance):
  Symptoms:
    - Low training error
    - High test error
    - Big gap: Training << Test
    
  Solutions:
    - Simplify model
    - Add regularization
    - Get more training data
    - Feature selection
    - Early stopping
    - Ensemble methods
""")

# Simulate train/test gap
from sklearn.model_selection import learning_curve

print("\nExample: Detecting overfitting with train vs test scores")
np.random.seed(42)
X = np.random.randn(500, 20)
y = (X[:, 0] + X[:, 1]**2 + np.random.randn(500) * 0.5) > 0.5

from sklearn.tree import DecisionTreeClassifier

print(f"{'Max Depth':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gap':>8}")
print("-" * 45)

for depth in [2, 5, 10, 20, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    train_scores = []
    test_scores = []
    
    from sklearn.model_selection import cross_val_score, ShuffleSplit
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    for train_idx, test_idx in cv.split(X):
        model.fit(X[train_idx], y[train_idx])
        train_scores.append(model.score(X[train_idx], y[train_idx]))
        test_scores.append(model.score(X[test_idx], y[test_idx]))
    
    train_acc = np.mean(train_scores)
    test_acc = np.mean(test_scores)
    gap = train_acc - test_acc
    
    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:>10} | {train_acc:>10.3f} | {test_acc:>10.3f} | {gap:>8.3f}")
```

## Learning Curves

```python
print("\n=== LEARNING CURVES ===")
print("""
Learning curves show how performance changes with training set size:

HIGH BIAS (Underfitting):
  - Training and validation errors both plateau HIGH
  - Adding more data won't help much
  - Need more complex model

    Error
      │
    High├──────────────  validation error
      │  ──────────────  training error
    Low └──────────────────────────────
              Training Set Size

HIGH VARIANCE (Overfitting):
  - Large gap between training and validation
  - More data may help (curves converging)

    Error
      │    ─ ─ ─ ─ ─ ─  validation error
      │              ╲
    Low├──────────────  training error
      └──────────────────────────────
              Training Set Size
""")
```

## Controlling Model Complexity

```python
print("\n=== METHODS TO CONTROL COMPLEXITY ===")
print("""
REDUCE COMPLEXITY (Less Variance):
  
  Linear Models:
    - Increase regularization (larger λ)
    - L1 (Lasso) for feature selection
    - L2 (Ridge) for shrinkage
  
  Decision Trees:
    - Decrease max_depth
    - Increase min_samples_split
    - Increase min_samples_leaf
  
  Random Forest:
    - Fewer trees (less overfitting risk)
    - Shallower trees
    - max_features < n_features
  
  Neural Networks:
    - Fewer layers/neurons
    - Dropout
    - Early stopping
    - Weight decay (L2)

INCREASE COMPLEXITY (Less Bias):
  
  - Opposite of above
  - Add polynomial features
  - Feature engineering
  - More powerful model family
""")
```

## Key Points

- **Bias**: Error from oversimplified models (underfitting)
- **Variance**: Error from oversensitivity to training data (overfitting)
- **Tradeoff**: Reducing one typically increases the other
- **Total error**: Bias² + Variance + Irreducible error
- **Diagnosis**: Compare train vs test error
- **Underfitting**: Both errors high, similar values
- **Overfitting**: Train error << Test error (big gap)
- **Goal**: Find optimal complexity that minimizes total error

## Reflection Questions

1. Why can't we simply minimize both bias and variance at the same time?
2. How does getting more training data affect bias and variance?
3. Why do ensemble methods like Random Forest reduce variance without increasing bias much?
