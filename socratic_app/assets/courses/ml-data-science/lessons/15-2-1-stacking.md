# Stacking and Model Ensembles

## Introduction

Stacking (stacked generalization) combines multiple models by training a meta-learner on their predictions. It's a powerful technique for squeezing extra performance from diverse models.

## Stacking Architecture

```python
import numpy as np
import pandas as pd

print("=== STACKING CONCEPT ===")
print("""
Two-level architecture:

LEVEL 0 (Base models):
  - Multiple different algorithms
  - Trained on original features
  - Generate predictions

LEVEL 1 (Meta-learner):
  - Trained on base model predictions
  - Learns to combine them optimally
  - Makes final prediction

    Original Features X
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
  Model1 Model2 Model3  ← Level 0 (base)
    ↓      ↓      ↓
    p1     p2     p3
    └──────┼──────┘
           ↓
      Meta-learner      ← Level 1 (meta)
           ↓
    Final Prediction
""")
```

## Why Stacking Works

```python
print("\n=== WHY STACKING WORKS ===")
print("""
Different models have different strengths:

Random Forest: Good with non-linear, handles noise
Linear Model: Captures linear relationships
Neural Network: Learns complex patterns
Gradient Boosting: Sequential error correction

Example predictions for 5 samples:

              Sample 1  Sample 2  Sample 3  Sample 4  Sample 5
True label:      1         0         1         1         0
RF:              1         0         0         1         0    ✓✓✗✓✓
LogReg:          0         0         1         0         0    ✗✓✓✗✓
XGBoost:         1         1         1         1         0    ✓✗✓✓✓
              
Each model: 80% accuracy
Majority vote: 5/5 = 100% accuracy!

Meta-learner can do even better than majority vote
by learning WHEN to trust each model.
""")
```

## Cross-Validation for Stacking

```python
print("\n=== AVOIDING DATA LEAKAGE ===")
print("""
WRONG: Train base models on all data, then train meta-learner
  - Meta-learner sees "too good" predictions
  - Base models already saw this data
  - Overfitting!

CORRECT: Use cross-validation for base predictions

1. Split data into K folds
2. For each fold:
   - Train base models on K-1 folds
   - Predict on held-out fold
3. Concatenate all predictions
4. Train meta-learner on these out-of-fold predictions

This gives "honest" predictions for meta-learner training.
""")

def generate_stacking_features(X, y, base_models, n_folds=5):
    """Generate meta-features using cross-validation"""
    from sklearn.model_selection import KFold
    
    n_samples = len(X)
    n_models = len(base_models)
    
    # Store out-of-fold predictions
    meta_features = np.zeros((n_samples, n_models))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        
        for model_idx, model in enumerate(base_models):
            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            # Predict on validation fold
            if hasattr(model_clone, 'predict_proba'):
                preds = model_clone.predict_proba(X_val)[:, 1]
            else:
                preds = model_clone.predict(X_val)
            
            meta_features[val_idx, model_idx] = preds
    
    return meta_features

print("Result: Each sample has predictions from models that NEVER saw it")
```

## Sklearn Implementation

```python
print("\n=== SKLEARN STACKINGCLASSIFIER ===")
print("""
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Define meta-learner
meta_learner = LogisticRegression()

# Create stacking classifier
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,                    # Cross-validation for meta-features
    stack_method='auto',     # Use predict_proba if available
    passthrough=False,       # Don't include original features
    n_jobs=-1
)

stacking.fit(X_train, y_train)
print(f"Stacking accuracy: {stacking.score(X_test, y_test):.3f}")

# With passthrough=True: meta-learner sees original features too
# [base_pred_1, base_pred_2, ..., original_features]
""")
```

## Blending

```python
print("\n=== BLENDING (SIMPLIFIED STACKING) ===")
print("""
Blending uses a hold-out set instead of cross-validation:

1. Split data: 70% train, 30% blend
2. Train base models on train set
3. Generate predictions on blend set
4. Train meta-learner on blend predictions

Advantages:
  - Simpler implementation
  - Faster (no cross-validation)
  
Disadvantages:
  - Less data for base models
  - Less data for meta-learner
  - Higher variance
""")

def blend_models(X_train, y_train, X_test, base_models, meta_model, blend_ratio=0.3):
    """Simple blending implementation"""
    from sklearn.model_selection import train_test_split
    
    # Split for blending
    X_tr, X_blend, y_tr, y_blend = train_test_split(
        X_train, y_train, test_size=blend_ratio, random_state=42
    )
    
    # Train base models and generate blend predictions
    blend_preds = []
    test_preds = []
    
    for model in base_models:
        model.fit(X_tr, y_tr)
        
        if hasattr(model, 'predict_proba'):
            blend_preds.append(model.predict_proba(X_blend)[:, 1])
            test_preds.append(model.predict_proba(X_test)[:, 1])
        else:
            blend_preds.append(model.predict(X_blend))
            test_preds.append(model.predict(X_test))
    
    blend_features = np.column_stack(blend_preds)
    test_features = np.column_stack(test_preds)
    
    # Train meta-learner
    meta_model.fit(blend_features, y_blend)
    
    # Final predictions
    final_preds = meta_model.predict(test_features)
    
    return final_preds

print("Blending is popular in competitions for its simplicity.")
```

## Weighted Averaging

```python
print("\n=== WEIGHTED AVERAGING ===")
print("""
Simpler alternative: weighted average of predictions

weights × predictions → final prediction

Finding optimal weights:
1. Grid search over weight combinations
2. Optimize weights to minimize validation loss
3. Use inverse of validation error
""")

def optimize_weights(predictions, y_true, n_iter=1000):
    """Find optimal weights for ensemble averaging"""
    n_models = predictions.shape[1]
    best_weights = None
    best_score = 0
    
    for _ in range(n_iter):
        # Random weights
        weights = np.random.random(n_models)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted prediction
        weighted_pred = np.dot(predictions, weights)
        
        # Score (assuming classification with threshold 0.5)
        pred_labels = (weighted_pred > 0.5).astype(int)
        score = np.mean(pred_labels == y_true)
        
        if score > best_score:
            best_score = score
            best_weights = weights
    
    return best_weights, best_score

# Example
predictions = np.array([
    [0.8, 0.7, 0.9],  # Sample 1
    [0.3, 0.4, 0.2],  # Sample 2
    [0.6, 0.5, 0.7],  # Sample 3
])
y_true = np.array([1, 0, 1])

weights, score = optimize_weights(predictions, y_true)
print(f"Optimal weights: {weights.round(3)}")
print(f"Ensemble score: {score:.3f}")
```

## Model Diversity

```python
print("\n=== DIVERSITY IS KEY ===")
print("""
Ensemble performance depends on:
1. Individual model accuracy
2. Model DIVERSITY (different errors)

Correlation between models:
  High correlation = Similar errors = Little benefit
  Low correlation = Different errors = Better ensemble

Ways to create diversity:
  - Different algorithms (RF, SVM, NN, ...)
  - Different hyperparameters
  - Different features subsets
  - Different training samples
  
Checking diversity:
  - Correlation of predictions
  - Diversity metrics (Q-statistic, disagreement)
""")

def check_model_diversity(predictions_dict):
    """Check correlation between model predictions"""
    df = pd.DataFrame(predictions_dict)
    corr = df.corr()
    
    print("Prediction correlations:")
    print(corr.round(2))
    
    # Average pairwise correlation
    n_models = len(predictions_dict)
    total_corr = 0
    count = 0
    for i in range(n_models):
        for j in range(i+1, n_models):
            total_corr += corr.iloc[i, j]
            count += 1
    
    avg_corr = total_corr / count if count > 0 else 0
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")
    print("Lower correlation = more diversity = better ensemble potential")

# Example
predictions_dict = {
    'RF': [0.8, 0.3, 0.6, 0.9, 0.2],
    'LogReg': [0.7, 0.4, 0.5, 0.8, 0.3],
    'SVM': [0.6, 0.2, 0.7, 0.5, 0.1],  # More different
}
check_model_diversity(predictions_dict)
```

## Multi-Level Stacking

```python
print("\n=== MULTI-LEVEL STACKING ===")
print("""
Stack multiple levels for complex ensembles:

Level 0: Base models (5-10 models)
Level 1: First meta-models (2-3 models)
Level 2: Final meta-model (1 model)

    ┌────┬────┬────┬────┬────┐
    │ M1 │ M2 │ M3 │ M4 │ M5 │  Level 0
    └─┬──┴─┬──┴─┬──┴─┬──┴─┬──┘
      └────┬────┴────┬────┘
      ┌────┴────┐┌───┴────┐
      │  Meta1  ││ Meta2  │     Level 1
      └────┬────┘└───┬────┘
           └────┬────┘
           ┌────┴────┐
           │ Final   │          Level 2
           └─────────┘

Caution:
  - Diminishing returns after 2 levels
  - More complex = harder to tune
  - Higher overfitting risk
  - Often overkill except in competitions
""")
```

## Key Points

- **Stacking**: Meta-learner combines base model predictions
- **Cross-validation**: Essential to avoid data leakage
- **Blending**: Simpler alternative using hold-out set
- **Diversity**: Different models = better ensemble
- **Weighted averaging**: Simple but effective combination
- **sklearn**: StackingClassifier handles the complexity

## Reflection Questions

1. Why is cross-validation necessary when generating meta-features?
2. How does model diversity affect ensemble performance?
3. When would weighted averaging be preferred over stacking?
