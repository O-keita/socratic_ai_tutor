# Evaluation Metrics for Classification

## Introduction

Accuracy alone is often insufficient for evaluating classification models. This lesson covers comprehensive metrics for understanding classifier performance, especially with imbalanced classes.

## The Confusion Matrix

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                           recall_score, f1_score, classification_report,
                           roc_curve, auc, precision_recall_curve)

np.random.seed(42)

print("=== CONFUSION MATRIX ===")
print("""
For binary classification (Positive = 1, Negative = 0):

                     PREDICTED
                  Negative  Positive
         Negative    TN        FP
ACTUAL   
         Positive    FN        TP

TN (True Negative): Correctly predicted negative
FP (False Positive): Incorrectly predicted positive (Type I error)
FN (False Negative): Incorrectly predicted negative (Type II error)
TP (True Positive): Correctly predicted positive
""")

# Example predictions
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
```

## Basic Metrics

```python
print("\n=== BASIC METRICS ===")

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy = (TP+TN)/(Total) = {accuracy:.3f}")

# Precision (Positive Predictive Value)
precision = tp / (tp + fp)
print(f"Precision = TP/(TP+FP) = {precision:.3f}")
print("  'Of those predicted positive, how many are correct?'")

# Recall (Sensitivity, True Positive Rate)
recall = tp / (tp + fn)
print(f"Recall = TP/(TP+FN) = {recall:.3f}")
print("  'Of actual positives, how many did we catch?'")

# Specificity (True Negative Rate)
specificity = tn / (tn + fp)
print(f"Specificity = TN/(TN+FP) = {specificity:.3f}")
print("  'Of actual negatives, how many did we correctly identify?'")

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score = 2*(Precision*Recall)/(Precision+Recall) = {f1:.3f}")
print("  Harmonic mean of precision and recall")
```

## When to Use Which Metric

```python
print("\n=== CHOOSING METRICS ===")
print("""
ACCURACY:
  Use when: Classes are balanced, equal cost of errors
  Problem: Misleading for imbalanced data
  Example: 95% accuracy on 95% majority class = useless

PRECISION:
  Maximize when: False positives are costly
  Examples:
    - Spam detection (don't block legitimate email)
    - Recommending products (don't annoy users)

RECALL:
  Maximize when: False negatives are costly
  Examples:
    - Cancer screening (don't miss disease)
    - Fraud detection (catch all fraud)
    - Search engines (find all relevant results)

F1 SCORE:
  Use when: Need balance between precision and recall
  When classes are imbalanced

SPECIFICITY:
  Important when: True negatives matter
  Examples:
    - Diagnostic tests (correctly identify healthy)
""")
```

## The Precision-Recall Tradeoff

```python
print("\n=== PRECISION-RECALL TRADEOFF ===")
print("""
By changing the decision threshold, you trade off precision and recall:

Higher threshold:
  - Fewer positive predictions
  - Higher precision (more confident)
  - Lower recall (miss more positives)

Lower threshold:
  - More positive predictions
  - Lower precision (more false positives)
  - Higher recall (catch more positives)
""")

# Simulated probabilities
np.random.seed(42)
y_true = np.array([0]*50 + [1]*50)
y_prob = np.concatenate([
    np.random.beta(2, 5, 50),  # Negative class (lower probs)
    np.random.beta(5, 2, 50)   # Positive class (higher probs)
])

thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
print("\nThreshold | Precision | Recall | F1")
print("-" * 45)
for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    print(f"  {thresh:.1f}     |   {p:.3f}   | {r:.3f}  | {f:.3f}")
```

## ROC Curve and AUC

```python
print("\n=== ROC CURVE AND AUC ===")
print("""
ROC (Receiver Operating Characteristic):
  - Plots True Positive Rate vs False Positive Rate
  - At different thresholds
  
  TPR (Recall) = TP / (TP + FN)
  FPR = FP / (FP + TN)

AUC (Area Under ROC Curve):
  - Ranges from 0 to 1
  - Random classifier = 0.5
  - Perfect classifier = 1.0
  - AUC = P(score(positive) > score(negative))
""")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print(f"AUC Score: {roc_auc:.3f}")

print("\nROC curve points (sample):")
print("Threshold | FPR   | TPR")
print("-" * 30)
for i in [0, len(thresholds)//4, len(thresholds)//2, -1]:
    if thresholds[i] != float('inf'):
        print(f"  {thresholds[i]:.3f}   | {fpr[i]:.3f} | {tpr[i]:.3f}")

print("""
Interpretation:
  - AUC 0.5: No discrimination (random)
  - AUC 0.7-0.8: Acceptable
  - AUC 0.8-0.9: Excellent  
  - AUC > 0.9: Outstanding
""")
```

## Precision-Recall Curve

```python
print("\n=== PRECISION-RECALL CURVE ===")
print("""
Alternative to ROC, especially for imbalanced data:
  - Plots Precision vs Recall at different thresholds
  - More informative when positive class is rare

Average Precision (AP):
  - Area under PR curve
  - Weighted mean of precision at each threshold
""")

precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_prob)

# Calculate average precision
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_true, y_prob)
print(f"Average Precision: {ap:.3f}")

print("\nWhy use PR curve over ROC?")
print("""
  - ROC can be overly optimistic for imbalanced data
  - PR focuses on positive class performance
  - Better when positives are rare and important
  - Example: Disease screening (few positives)
""")
```

## Multiclass Metrics

```python
print("\n=== MULTICLASS METRICS ===")
print("""
For K classes, calculate metrics per class then average:

MACRO AVERAGE:
  - Calculate metric for each class
  - Average them (equal weight to each class)
  - Good when classes equally important

MICRO AVERAGE:
  - Pool all predictions
  - Calculate metric on pooled data
  - Weights classes by frequency

WEIGHTED AVERAGE:
  - Average weighted by class support (count)
  - Accounts for class imbalance
""")

# Example multiclass
y_true_mc = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
y_pred_mc = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2, 0])

print("Classification Report:")
print(classification_report(y_true_mc, y_pred_mc, 
                           target_names=['Class 0', 'Class 1', 'Class 2']))

# Manual calculation
print("\nMacro vs Micro:")
print(f"  Macro F1: {f1_score(y_true_mc, y_pred_mc, average='macro'):.3f}")
print(f"  Micro F1: {f1_score(y_true_mc, y_pred_mc, average='micro'):.3f}")
print(f"  Weighted F1: {f1_score(y_true_mc, y_pred_mc, average='weighted'):.3f}")
```

## Handling Imbalanced Classes

```python
print("\n=== IMBALANCED CLASSES ===")
print("""
When one class dominates (e.g., 95% negative, 5% positive):

PROBLEMS:
  - Accuracy is misleading (95% by predicting all negative!)
  - Model biased toward majority class

SOLUTIONS:
  1. Use appropriate metrics (F1, AUC, PR-AUC)
  2. Resample data (oversample minority, undersample majority)
  3. Use class weights in model
  4. Use stratified cross-validation
  5. Set appropriate threshold
""")

# Example with imbalanced data
y_true_imb = np.array([0]*95 + [1]*5)
y_pred_imb = np.array([0]*100)  # Predicts all negative

print("Highly imbalanced example (95% negative):")
print(f"  Accuracy: {accuracy_score(y_true_imb, y_pred_imb):.3f}")
print(f"  Precision: {precision_score(y_true_imb, y_pred_imb, zero_division=0):.3f}")
print(f"  Recall: {recall_score(y_true_imb, y_pred_imb):.3f}")
print(f"  F1: {f1_score(y_true_imb, y_pred_imb):.3f}")
print("\n  High accuracy but useless model!")
```

## Summary Table

```python
print("\n=== METRICS SUMMARY ===")
print("""
METRIC          FORMULA                  USE WHEN
─────────────────────────────────────────────────────────────
Accuracy        (TP+TN)/Total            Balanced classes
Precision       TP/(TP+FP)               FP is costly
Recall          TP/(TP+FN)               FN is costly
Specificity     TN/(TN+FP)               TN is important
F1 Score        2*P*R/(P+R)              Balance P and R
AUC-ROC         Area under ROC           Ranking ability
PR-AUC          Area under PR            Imbalanced data

REMEMBER:
  - No single metric tells the whole story
  - Consider the business context
  - Always look at the confusion matrix
  - Use multiple metrics
""")
```

## Key Points

- **Confusion matrix**: Foundation for all metrics
- **Accuracy**: Misleading for imbalanced data
- **Precision**: Minimize false positives
- **Recall**: Minimize false negatives (catch all positives)
- **F1 Score**: Balance precision and recall
- **ROC-AUC**: Overall ranking ability, threshold-independent
- **PR-AUC**: Better for imbalanced datasets
- **Threshold**: Adjust based on business requirements

## Reflection Questions

1. Why is accuracy not a good metric for fraud detection (where fraud is rare)?
2. How would you set the decision threshold for a cancer screening test?
3. When would micro-average be preferred over macro-average?
