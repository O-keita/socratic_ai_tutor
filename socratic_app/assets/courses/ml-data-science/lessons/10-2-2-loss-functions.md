# Loss Functions

## Introduction

Loss functions measure how well a neural network's predictions match the targets. The choice of loss function depends on the task and directly impacts what the model learns.

## What is a Loss Function?

```python
import numpy as np
import pandas as pd

print("=== LOSS FUNCTIONS ===")
print("""
LOSS FUNCTION: Measures prediction error

Loss = f(y_true, y_pred)

Properties:
  - Lower is better
  - Differentiable (for gradient descent)
  - Matches the task (classification vs regression)

Training minimizes average loss over all samples.
The gradient of loss drives weight updates.
""")
```

## Mean Squared Error (MSE)

```python
print("\n=== MEAN SQUARED ERROR ===")
print("""
MSE = (1/n) Σ(y_true - y_pred)²

Used for: REGRESSION tasks

Properties:
  - Penalizes large errors heavily (squared)
  - Always positive
  - Units are squared (e.g., dollars² for price prediction)
""")

def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    """Gradient of MSE w.r.t. y_pred"""
    return 2 * (y_pred - y_true) / len(y_true)

# Example
y_true = np.array([3.0, 5.0, 2.0, 8.0])
y_pred = np.array([2.5, 5.5, 2.0, 7.0])

loss = mse_loss(y_true, y_pred)
grad = mse_gradient(y_true, y_pred)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MSE: {loss:.4f}")
print(f"Gradient: {grad.round(4)}")

# Effect of large errors
y_pred_outlier = np.array([2.5, 5.5, 2.0, 3.0])  # Big error on last
print(f"\nWith outlier: MSE = {mse_loss(y_true, y_pred_outlier):.4f}")
print("Note: One large error dominates!")
```

## Mean Absolute Error (MAE)

```python
print("\n=== MEAN ABSOLUTE ERROR ===")
print("""
MAE = (1/n) Σ|y_true - y_pred|

Used for: REGRESSION (robust to outliers)

Properties:
  - Linear penalty (not squared)
  - Less sensitive to outliers
  - Not differentiable at zero (use smooth version)
""")

def mae_loss(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def smooth_mae_gradient(y_true, y_pred, eps=1e-8):
    """Gradient of smoothed MAE"""
    diff = y_pred - y_true
    return diff / (np.abs(diff) + eps) / len(y_true)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MAE: {mae_loss(y_true, y_pred):.4f}")

print(f"\nWith outlier:")
print(f"  MSE: {mse_loss(y_true, y_pred_outlier):.4f}")
print(f"  MAE: {mae_loss(y_true, y_pred_outlier):.4f}")
print("MAE is less affected by outliers!")
```

## Huber Loss

```python
print("\n=== HUBER LOSS ===")
print("""
Huber loss combines MSE and MAE:

L(δ) = {
  0.5 × (y_true - y_pred)²        if |error| ≤ δ
  δ × |y_true - y_pred| - 0.5δ²   if |error| > δ
}

Benefits:
  - Quadratic for small errors (like MSE)
  - Linear for large errors (like MAE)
  - Robust to outliers, still smooth
""")

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss"""
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    
    squared = 0.5 * error ** 2
    linear = delta * np.abs(error) - 0.5 * delta ** 2
    
    return np.mean(np.where(is_small, squared, linear))

print(f"Normal predictions:")
print(f"  MSE: {mse_loss(y_true, y_pred):.4f}")
print(f"  MAE: {mae_loss(y_true, y_pred):.4f}")
print(f"  Huber: {huber_loss(y_true, y_pred):.4f}")

print(f"\nWith outlier:")
print(f"  MSE: {mse_loss(y_true, y_pred_outlier):.4f}")
print(f"  MAE: {mae_loss(y_true, y_pred_outlier):.4f}")
print(f"  Huber: {huber_loss(y_true, y_pred_outlier):.4f}")
```

## Binary Cross-Entropy

```python
print("\n=== BINARY CROSS-ENTROPY ===")
print("""
BCE = -(1/n) Σ[y×log(p) + (1-y)×log(1-p)]

Used for: BINARY CLASSIFICATION

Where:
  - y ∈ {0, 1} is true label
  - p ∈ (0, 1) is predicted probability

Properties:
  - Penalizes confident wrong predictions heavily
  - Works with sigmoid output
  - Also called "log loss"
""")

def binary_crossentropy(y_true, y_pred, eps=1e-15):
    """Binary cross-entropy loss"""
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example: Binary classification
y_true_bin = np.array([1, 1, 0, 0])
y_pred_good = np.array([0.9, 0.8, 0.2, 0.1])  # Good predictions
y_pred_bad = np.array([0.1, 0.2, 0.8, 0.9])   # Bad predictions
y_pred_conf = np.array([0.99, 0.99, 0.01, 0.01])  # Confident correct

print(f"y_true: {y_true_bin}")
print(f"\nGood predictions: {y_pred_good}")
print(f"  BCE: {binary_crossentropy(y_true_bin, y_pred_good):.4f}")

print(f"\nBad predictions: {y_pred_bad}")
print(f"  BCE: {binary_crossentropy(y_true_bin, y_pred_bad):.4f}")

print(f"\nConfident correct: {y_pred_conf}")
print(f"  BCE: {binary_crossentropy(y_true_bin, y_pred_conf):.4f}")
```

## Categorical Cross-Entropy

```python
print("\n=== CATEGORICAL CROSS-ENTROPY ===")
print("""
CCE = -(1/n) Σ Σ y_true[c] × log(y_pred[c])

Used for: MULTI-CLASS CLASSIFICATION

Where:
  - y_true is one-hot encoded [0, 1, 0]
  - y_pred is softmax output [0.1, 0.7, 0.2]

Sum over all classes, average over samples.
""")

def softmax(logits):
    """Softmax function"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def categorical_crossentropy(y_true, y_pred, eps=1e-15):
    """Categorical cross-entropy"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

# Example: 3-class classification
y_true_cat = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
])

y_pred_cat = softmax(np.array([
    [2.0, 0.5, 0.3],  # Predicts class 0
    [0.2, 2.5, 0.4],  # Predicts class 1
    [0.1, 0.3, 3.0],  # Predicts class 2
]))

print(f"True labels (one-hot):")
print(y_true_cat)
print(f"\nPredicted probabilities:")
print(y_pred_cat.round(3))
print(f"\nCategorical Cross-Entropy: {categorical_crossentropy(y_true_cat, y_pred_cat):.4f}")
```

## Sparse Categorical Cross-Entropy

```python
print("\n=== SPARSE CATEGORICAL CROSS-ENTROPY ===")
print("""
Same as categorical CE, but labels are integers, not one-hot:

y_true = [0, 1, 2]  instead of [[1,0,0], [0,1,0], [0,0,1]]

More memory efficient for many classes.
Common in frameworks: sparse_categorical_crossentropy
""")

def sparse_categorical_crossentropy(y_true, y_pred, eps=1e-15):
    """Sparse categorical cross-entropy"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n_samples = len(y_true)
    # Select the predicted probability for the true class
    return -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))

y_true_sparse = np.array([0, 1, 2])  # Integer labels

print(f"Sparse labels: {y_true_sparse}")
print(f"Predictions: {y_pred_cat.round(3)}")
print(f"Sparse CCE: {sparse_categorical_crossentropy(y_true_sparse, y_pred_cat):.4f}")
```

## Loss Function Summary

```python
print("\n=== LOSS FUNCTION SUMMARY ===")
print("""
REGRESSION:
  - MSE: Standard choice, sensitive to outliers
  - MAE: Robust to outliers, not smooth at 0
  - Huber: Best of both worlds
  - MAPE: Percentage errors (scale independent)

BINARY CLASSIFICATION:
  - Binary Cross-Entropy: Standard choice
  - Hinge Loss: For SVMs (margin-based)
  - Focal Loss: For imbalanced data

MULTI-CLASS CLASSIFICATION:
  - Categorical CE: One-hot labels
  - Sparse Categorical CE: Integer labels

SPECIAL CASES:
  - Contrastive Loss: Siamese networks
  - Triplet Loss: Embedding learning
  - KL Divergence: Distribution matching
""")

# Quick reference table
print("\nQuick Reference:")
print(f"{'Task':<25} {'Loss Function':<30} {'Output Activation':<20}")
print("-" * 75)
print(f"{'Regression':<25} {'MSE, MAE, Huber':<30} {'Linear (none)':<20}")
print(f"{'Binary Classification':<25} {'Binary Cross-Entropy':<30} {'Sigmoid':<20}")
print(f"{'Multi-class':<25} {'Categorical Cross-Entropy':<30} {'Softmax':<20}")
print(f"{'Multi-label':<25} {'Binary CE per label':<30} {'Sigmoid (each)':<20}")
```

## Key Points

- **MSE**: Regression, penalizes large errors heavily
- **MAE**: Regression, robust to outliers
- **Huber**: Combines MSE and MAE benefits
- **Binary CE**: Binary classification with sigmoid
- **Categorical CE**: Multi-class with softmax
- **Match loss to task**: Wrong loss = wrong optimization target
- **Output activation must match loss function**

## Reflection Questions

1. Why does cross-entropy penalize confident wrong predictions more than uncertain ones?
2. When would you prefer MAE over MSE for a regression task?
3. How does the choice of loss function affect what the model learns to optimize?
