# Handling Imbalanced Data

## Introduction

Many real-world classification problems have imbalanced class distributions (e.g., fraud detection, disease diagnosis). Standard algorithms often struggle with imbalanced data, requiring special techniques.

## The Imbalanced Data Problem

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, precision_recall_curve, average_precision_score)

np.random.seed(42)

print("=== THE IMBALANCED DATA PROBLEM ===")
print("""
Examples of imbalanced problems:
  - Fraud detection: 0.1% fraudulent transactions
  - Medical diagnosis: 1% have the disease
  - Manufacturing defects: 0.5% defective items
  - Churn prediction: 5% customers churn

Problems with standard approaches:
  1. Accuracy is misleading (99% by predicting majority)
  2. Model biased toward majority class
  3. Minority class (often the important one!) poorly predicted
""")

# Create imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                          n_redundant=5, weights=[0.95, 0.05], 
                          flip_y=0.01, random_state=42)

print(f"\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     stratify=y, random_state=42)
```

## Baseline: Standard Approach

```python
print("\n=== BASELINE: STANDARD CLASSIFIER ===")

# Standard logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))

print("""
Notice:
  - High overall accuracy (but misleading!)
  - Minority class recall is low (missing many positives)
  - The model is biased toward predicting majority class
""")
```

## Evaluation Metrics for Imbalanced Data

```python
print("\n=== BETTER METRICS FOR IMBALANCED DATA ===")
print("""
DON'T USE: Accuracy alone

DO USE:
  1. PRECISION: Of predicted positives, how many correct?
  2. RECALL: Of actual positives, how many found?
  3. F1-SCORE: Balance of precision and recall
  4. ROC-AUC: Ranking ability (threshold-independent)
  5. PR-AUC: Better for severe imbalance

Focus on the MINORITY CLASS metrics!
""")

y_prob = model.predict_proba(X_test)[:, 1]

print("Minority class metrics:")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"  PR-AUC: {average_precision_score(y_test, y_prob):.3f}")

# Precision-Recall at different thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
print(f"\nPrecision-Recall tradeoff (sample):")
for thresh in [0.3, 0.5, 0.7]:
    idx = np.argmin(np.abs(thresholds - thresh))
    print(f"  Threshold {thresh}: Precision={precision[idx]:.3f}, Recall={recall[idx]:.3f}")
```

## Solution 1: Class Weights

```python
print("\n=== SOLUTION 1: CLASS WEIGHTS ===")
print("""
Penalize misclassifying minority class more heavily.
Effectively makes minority examples "worth more".

In sklearn: class_weight parameter
  - 'balanced': Weights inversely proportional to class frequencies
  - Custom dict: {class: weight}
""")

# Balanced class weights
model_weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print("With class_weight='balanced':")
print(classification_report(y_test, y_pred_weighted, target_names=['Majority', 'Minority']))

# Custom weights
print("Effect of different class weights on minority class:")
for weight in [1, 5, 10, 20]:
    model_w = LogisticRegression(class_weight={0: 1, 1: weight}, max_iter=1000)
    model_w.fit(X_train, y_train)
    y_pred_w = model_w.predict(X_test)
    recall = np.sum((y_pred_w == 1) & (y_test == 1)) / np.sum(y_test == 1)
    precision = np.sum((y_pred_w == 1) & (y_test == 1)) / np.sum(y_pred_w == 1) if np.sum(y_pred_w == 1) > 0 else 0
    print(f"  Weight {weight:2d}: Recall={recall:.3f}, Precision={precision:.3f}")
```

## Solution 2: Resampling

```python
print("\n=== SOLUTION 2: RESAMPLING ===")
print("""
OVERSAMPLING (Minority):
  - Duplicate minority class examples
  - Or generate synthetic examples (SMOTE)
  - Risk: Overfitting to minority examples

UNDERSAMPLING (Majority):
  - Remove majority class examples
  - Risk: Losing valuable information

COMBINATION:
  - Undersample majority + Oversample minority
""")

# Manual random oversampling
minority_idx = np.where(y_train == 1)[0]
majority_idx = np.where(y_train == 0)[0]

# Oversample minority to match majority
n_majority = len(majority_idx)
minority_oversampled = np.random.choice(minority_idx, size=n_majority, replace=True)

X_train_over = np.vstack([X_train[majority_idx], X_train[minority_oversampled]])
y_train_over = np.concatenate([y_train[majority_idx], y_train[minority_oversampled]])

print(f"Original: {np.bincount(y_train)}")
print(f"After oversampling: {np.bincount(y_train_over)}")

model_over = LogisticRegression(max_iter=1000)
model_over.fit(X_train_over, y_train_over)
y_pred_over = model_over.predict(X_test)

print("\nWith random oversampling:")
print(classification_report(y_test, y_pred_over, target_names=['Majority', 'Minority']))
```

## Solution 3: SMOTE

```python
print("\n=== SOLUTION 3: SMOTE ===")
print("""
Synthetic Minority Over-sampling Technique (SMOTE):
  1. For each minority example
  2. Find k nearest neighbors (in minority class)
  3. Create synthetic example between point and neighbor

Better than simple duplication:
  - Creates NEW examples
  - Expands decision region
  - Less overfitting

Variants:
  - SMOTE-NC: For mixed continuous/categorical
  - Borderline-SMOTE: Focus on border examples
  - ADASYN: Adaptive synthetic sampling
""")

# Demonstrate SMOTE concept (manual simple version)
def simple_smote(X_minority, n_synthetic, k=5):
    """Simplified SMOTE for demonstration"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_minority)
    
    synthetic = []
    for _ in range(n_synthetic):
        # Random minority example
        idx = np.random.randint(len(X_minority))
        x = X_minority[idx]
        
        # Random neighbor
        distances, indices = nn.kneighbors([x])
        neighbor_idx = np.random.choice(indices[0][1:])  # Exclude self
        neighbor = X_minority[neighbor_idx]
        
        # Interpolate
        alpha = np.random.random()
        synthetic.append(x + alpha * (neighbor - x))
    
    return np.array(synthetic)

# Apply simple SMOTE
X_minority = X_train[y_train == 1]
n_to_generate = len(majority_idx) - len(minority_idx)
X_synthetic = simple_smote(X_minority, n_to_generate)

X_train_smote = np.vstack([X_train, X_synthetic])
y_train_smote = np.concatenate([y_train, np.ones(n_to_generate)])

print(f"After SMOTE: {np.bincount(y_train_smote.astype(int))}")

model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)

print("\nWith SMOTE:")
print(classification_report(y_test, y_pred_smote, target_names=['Majority', 'Minority']))
```

## Solution 4: Threshold Adjustment

```python
print("\n=== SOLUTION 4: THRESHOLD ADJUSTMENT ===")
print("""
Instead of default threshold (0.5), choose optimal threshold:
  - Lower threshold: More positive predictions (higher recall)
  - Higher threshold: Fewer positive predictions (higher precision)

Find threshold that maximizes F1 or achieves desired recall.
""")

# Find optimal threshold for F1
from sklearn.metrics import f1_score

y_prob = model.predict_proba(X_test)[:, 1]

best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_prob >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Optimal threshold for F1: {best_thresh:.2f} (F1={best_f1:.3f})")

# Compare with default
y_pred_default = model.predict(X_test)
y_pred_optimal = (y_prob >= best_thresh).astype(int)

print("\nDefault threshold (0.5) vs Optimal:")
print(f"  Default: F1={f1_score(y_test, y_pred_default):.3f}")
print(f"  Optimal: F1={f1_score(y_test, y_pred_optimal):.3f}")
```

## Ensemble Methods for Imbalanced Data

```python
print("\n=== ENSEMBLE METHODS ===")
print("""
BAGGING WITH BALANCED SUBSETS:
  - Train multiple models
  - Each on balanced bootstrap sample

BOOSTING WITH SAMPLE WEIGHTS:
  - Misclassified minority examples get higher weight
  - AdaBoost, Gradient Boosting can help

BALANCED RANDOM FOREST:
  - Sample balanced subset for each tree
  - Often very effective
""")

from sklearn.ensemble import RandomForestClassifier

# Random Forest with balanced subsampling
rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample',
                                     random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_rf = rf_balanced.predict(X_test)

print("Random Forest with balanced_subsample:")
print(classification_report(y_test, y_pred_rf, target_names=['Majority', 'Minority']))
```

## Summary of Approaches

```python
print("\n=== SUMMARY OF APPROACHES ===")
print("""
APPROACH           | PROS                    | CONS
───────────────────┼─────────────────────────┼─────────────────────
Class Weights      | Simple, no data change  | May not be enough
Random Oversample  | Simple                  | Overfitting risk
Random Undersample | Simple, faster training | Loses information
SMOTE              | Creates new examples    | Can create noise
Threshold Tuning   | No retraining needed    | Needs probability output
Ensemble Methods   | Often most effective    | More complex

RECOMMENDATIONS:
1. Always use stratified cross-validation
2. Evaluate with appropriate metrics (F1, AUC, PR-AUC)
3. Start with class_weight='balanced'
4. Try SMOTE if class_weight insufficient
5. Consider ensemble methods for best results
6. Tune decision threshold last
""")
```

## Key Points

- **Imbalanced data**: One class dominates, standard methods fail
- **Don't use accuracy**: Use F1, ROC-AUC, PR-AUC instead
- **Class weights**: Simple and often effective first step
- **Resampling**: Oversample minority or undersample majority
- **SMOTE**: Generate synthetic minority examples
- **Threshold tuning**: Adjust decision boundary
- **Ensembles**: Balanced subsampling often best
- **Stratified CV**: Always preserve class ratios in folds

## Reflection Questions

1. Why is accuracy misleading for a dataset with 99% negative and 1% positive examples?
2. When might undersampling be preferred over oversampling?
3. How would you choose between optimizing for precision vs recall in a fraud detection system?
