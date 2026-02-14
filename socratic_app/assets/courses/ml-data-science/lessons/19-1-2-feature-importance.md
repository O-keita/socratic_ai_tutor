# Feature Importance and Interpretability

## Introduction

Interpretability helps us understand why models make predictions. This is crucial for trust, debugging, compliance, and scientific discovery.

## Why Interpretability Matters

```python
import numpy as np
import pandas as pd

print("=== WHY INTERPRETABILITY? ===")
print("""
1. TRUST:
   - Do users believe the model?
   - High-stakes: Healthcare, finance, legal

2. DEBUGGING:
   - Is the model learning right patterns?
   - Detect spurious correlations

3. COMPLIANCE:
   - GDPR: "Right to explanation"
   - Fair lending laws

4. IMPROVEMENT:
   - Understand failure modes
   - Feature engineering insights

5. SCIENTIFIC DISCOVERY:
   - What does the model reveal about data?
   - Generate hypotheses
""")
```

## Model-Specific vs Model-Agnostic

```python
print("\n=== INTERPRETABILITY APPROACHES ===")
print("""
MODEL-SPECIFIC:
  Built into particular models
  - Linear regression: Coefficients
  - Decision trees: Rules
  - Attention weights
  
MODEL-AGNOSTIC:
  Works with any model
  - Permutation importance
  - SHAP values
  - LIME
  - Partial dependence plots

LOCAL vs GLOBAL:
  Local: Explain single prediction
  Global: Explain overall model behavior
""")
```

## Linear Model Coefficients

```python
print("\n=== LINEAR MODEL INTERPRETATION ===")
print("""
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Interpretation:
  - βᵢ: Change in y for 1-unit increase in xᵢ
  - Sign: Direction of relationship
  - Magnitude: Strength of relationship

IMPORTANT: Scale features first!
  - Otherwise coefficients not comparable
  - Use StandardScaler
""")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Simulated data
np.random.seed(42)
X = np.random.randn(100, 4)
y = 2*X[:, 0] - 0.5*X[:, 1] + 0.1*X[:, 2] + 0.8*X[:, 3] + np.random.randn(100)*0.1

# Scale and fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

print("Feature coefficients (standardized):")
for i, coef in enumerate(model.coef_):
    print(f"  Feature {i}: {coef:.3f}")

print("\nInterpretation:")
print("  Feature 0 has strongest positive effect")
print("  Feature 1 has negative effect")
```

## Permutation Importance

```python
print("\n=== PERMUTATION IMPORTANCE ===")
print("""
Model-agnostic method for any model

Algorithm:
1. Compute baseline model score
2. For each feature:
   a. Shuffle feature values (break relationship with target)
   b. Compute new model score
   c. Importance = baseline_score - shuffled_score
3. Repeat multiple times for stability

If feature important:
  - Shuffling hurts performance a lot
  - High importance score

If feature unimportant:
  - Shuffling doesn't change much
  - Low importance score
""")

def permutation_importance(model, X, y, n_repeats=10):
    """Compute permutation importance"""
    baseline_score = model.score(X, y)
    importances = []
    
    for feature_idx in range(X.shape[1]):
        scores = []
        
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            score = model.score(X_permuted, y)
            scores.append(baseline_score - score)
        
        importances.append({
            'mean': np.mean(scores),
            'std': np.std(scores)
        })
    
    return importances

importances = permutation_importance(model, X_scaled, y, n_repeats=5)

print("Permutation importance:")
for i, imp in enumerate(importances):
    print(f"  Feature {i}: {imp['mean']:.4f} ± {imp['std']:.4f}")
```

## SHAP Values

```python
print("\n=== SHAP VALUES ===")
print("""
SHAP = SHapley Additive exPlanations

Based on game theory (Shapley values):
  - Fairly distribute "payout" among players
  - Here: Distribute prediction among features

For prediction f(x):
  f(x) = E[f(X)] + Σ φᵢ
  
  E[f(X)]: Expected prediction (baseline)
  φᵢ: SHAP value for feature i

Properties:
  1. Local accuracy: Sum of SHAP values = prediction - baseline
  2. Missingness: Missing feature has 0 contribution
  3. Consistency: If feature contributes more → higher SHAP

Computation:
  - Exact: Exponential in features (tractable for trees)
  - Approximation: Sampling-based (KernelSHAP)
""")

print("""
Using SHAP library:

import shap

# Create explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# Summary plot (global importance)
shap.summary_plot(shap_values, X_test)

# Force plot (single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

# Dependence plot (feature effect)
shap.dependence_plot("feature_name", shap_values, X_test)
""")

# Simplified SHAP demonstration
def simplified_shap(model, x, baseline, n_samples=100):
    """Simplified SHAP approximation"""
    n_features = len(x)
    shap_values = np.zeros(n_features)
    
    for i in range(n_features):
        marginal_contributions = []
        
        for _ in range(n_samples):
            # Random subset of other features
            subset = np.random.randint(0, 2, n_features)
            subset[i] = 0
            
            # Prediction without feature i
            x_without = baseline.copy()
            x_without[subset == 1] = x[subset == 1]
            pred_without = model.predict(x_without.reshape(1, -1))[0]
            
            # Prediction with feature i
            x_with = x_without.copy()
            x_with[i] = x[i]
            pred_with = model.predict(x_with.reshape(1, -1))[0]
            
            marginal_contributions.append(pred_with - pred_without)
        
        shap_values[i] = np.mean(marginal_contributions)
    
    return shap_values

print("SHAP provides theoretically grounded feature attributions")
```

## Partial Dependence Plots

```python
print("\n=== PARTIAL DEPENDENCE PLOTS ===")
print("""
Show marginal effect of a feature on prediction

Algorithm:
1. For each value v of feature i:
   a. Set feature i = v for ALL samples
   b. Compute average prediction
2. Plot v vs average prediction

Shows: Average effect of feature on prediction
Hides: Interactions with other features

For 2 features: 2D heatmap showing interaction
""")

def partial_dependence(model, X, feature_idx, grid_points=50):
    """Compute partial dependence"""
    feature_values = np.linspace(X[:, feature_idx].min(), 
                                  X[:, feature_idx].max(), 
                                  grid_points)
    pd_values = []
    
    for value in feature_values:
        X_copy = X.copy()
        X_copy[:, feature_idx] = value
        avg_prediction = model.predict(X_copy).mean()
        pd_values.append(avg_prediction)
    
    return feature_values, pd_values

# Example
feature_values, pd_values = partial_dependence(model, X_scaled, feature_idx=0)

print("Partial dependence (Feature 0):")
print("  Feature value range:", f"{feature_values[0]:.2f} to {feature_values[-1]:.2f}")
print("  Prediction range:", f"{min(pd_values):.2f} to {max(pd_values):.2f}")
print("  This shows how predictions change as Feature 0 varies")
```

## LIME (Local Interpretable Model-agnostic Explanations)

```python
print("\n=== LIME ===")
print("""
Explain single prediction with local linear model

Algorithm:
1. For prediction to explain (x):
   a. Generate perturbed samples around x
   b. Get predictions from black-box model
   c. Weight samples by distance to x
   d. Fit simple model (linear) to weighted samples
   e. Interpret simple model

The simple model approximates black-box LOCALLY
  - Valid near x
  - Not globally accurate

Output: Weights showing feature contributions
""")

print("""
Using LIME library:

from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['negative', 'positive'],
    mode='classification'
)

# Explain single prediction
exp = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=5
)

# Show explanation
exp.show_in_notebook()
# Or
exp.as_list()  # [(feature, weight), ...]
""")

print("""
LIME vs SHAP:

LIME:
  - Local approximation
  - Faster
  - Less theoretically grounded

SHAP:
  - Based on Shapley values
  - Consistent, additive
  - More computationally expensive
  - Better theoretical properties
""")
```

## Key Points

- **Model-specific**: Use model structure (coefficients, trees)
- **Permutation importance**: Shuffle feature, measure impact
- **SHAP**: Game-theoretic fair attribution of prediction
- **PDP**: Show average feature effect
- **LIME**: Local linear approximation for single predictions
- **Trade-offs**: Accuracy vs interpretability, global vs local

## Reflection Questions

1. When would you prefer SHAP over permutation importance?
2. Why is scaling important for interpreting linear model coefficients?
3. What are the limitations of explaining complex models with simple approximations?
