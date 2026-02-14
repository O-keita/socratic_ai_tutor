# Model-Agnostic Explanations

## Introduction

Model-agnostic explanation methods work with any machine learning model, treating it as a black box. These techniques are essential for understanding complex models like neural networks and ensembles.

## The Black Box Problem

```python
import numpy as np
import pandas as pd

print("=== WHY MODEL-AGNOSTIC? ===")
print("""
PROBLEM: Complex models are black boxes
  - Neural networks: Millions of parameters
  - Ensembles: 100s of trees combined
  - Can't directly inspect decision logic

SOLUTION: Probe the model's behavior
  - Input perturbations
  - Output analysis
  - No access to internals needed

ADVANTAGES:
  ✓ Works with ANY model
  ✓ Compare explanations across models
  ✓ Apply to deployed models (API access)
  ✓ No implementation required

DISADVANTAGES:
  ✗ Can be computationally expensive
  ✗ Approximations may be inaccurate
  ✗ Different methods may give different explanations
""")
```

## Perturbation-Based Methods

```python
print("\n=== PERTURBATION METHODS ===")
print("""
Core idea: Modify inputs, observe output changes

Types of perturbations:
1. FEATURE REMOVAL: Set to zero/baseline
2. FEATURE SHUFFLING: Random permutation
3. FEATURE MASKING: Replace with mean/mode
4. NOISE INJECTION: Add Gaussian noise
5. SAMPLING: Generate similar instances

The change in prediction reveals feature importance.
""")

def occlusion_importance(model, X, baseline=None):
    """Feature importance via occlusion (masking)"""
    if baseline is None:
        baseline = np.zeros(X.shape[1])
    
    original_pred = model.predict(X.reshape(1, -1))[0]
    importances = []
    
    for i in range(len(X)):
        X_masked = X.copy()
        X_masked[i] = baseline[i]
        masked_pred = model.predict(X_masked.reshape(1, -1))[0]
        importances.append(original_pred - masked_pred)
    
    return np.array(importances)

# Demonstration
np.random.seed(42)
X_sample = np.array([1.5, -0.5, 0.2, 0.8])

# Mock model
class MockModel:
    def predict(self, X):
        return 2*X[:, 0] - 0.5*X[:, 1] + 0.1*X[:, 2] + 0.8*X[:, 3]

model = MockModel()
importance = occlusion_importance(model, X_sample)

print("Occlusion importance:")
for i, imp in enumerate(importance):
    print(f"  Feature {i}: {imp:.3f}")
```

## Counterfactual Explanations

```python
print("\n=== COUNTERFACTUAL EXPLANATIONS ===")
print("""
"What minimal change would alter the prediction?"

Counterfactual: Alternative instance where:
  - Prediction changes (e.g., reject → accept)
  - Minimal changes from original

Example (loan application):
  Original: Income=$50K, Age=25 → REJECTED
  Counterfactual: Income=$65K, Age=25 → ACCEPTED
  
  Explanation: "If income were $65K, loan approved"

BENEFITS:
  - Actionable insights
  - Human-understandable
  - Specific recommendations

CHALLENGES:
  - Finding valid counterfactuals
  - Multiple valid counterfactuals exist
  - Feasibility constraints
""")

def find_counterfactual(model, x, target_class, 
                        feature_ranges, max_iter=1000, step_size=0.1):
    """Simple gradient-based counterfactual search"""
    x_cf = x.copy()
    
    for _ in range(max_iter):
        # Current prediction
        pred = model.predict_proba(x_cf.reshape(1, -1))[0]
        
        if np.argmax(pred) == target_class:
            break
        
        # Move toward target (gradient approximation)
        for i in range(len(x_cf)):
            # Try increase
            x_plus = x_cf.copy()
            x_plus[i] += step_size
            pred_plus = model.predict_proba(x_plus.reshape(1, -1))[0][target_class]
            
            # Try decrease
            x_minus = x_cf.copy()
            x_minus[i] -= step_size
            pred_minus = model.predict_proba(x_minus.reshape(1, -1))[0][target_class]
            
            # Move in better direction
            if pred_plus > pred_minus:
                x_cf[i] = np.clip(x_plus[i], *feature_ranges[i])
            else:
                x_cf[i] = np.clip(x_minus[i], *feature_ranges[i])
    
    return x_cf

print("Counterfactuals provide 'what-if' explanations")
```

## Individual Conditional Expectation (ICE)

```python
print("\n=== ICE PLOTS ===")
print("""
ICE: Individual curves, not just averages

PDP shows AVERAGE effect → hides heterogeneity
ICE shows INDIVIDUAL effect for each instance

ICE Plot:
  - One line per instance
  - Shows how prediction changes with feature
  - Reveals interactions and heterogeneous effects

If ICE lines parallel: No interaction with other features
If ICE lines cross: Interaction exists
""")

def ice_curves(model, X, feature_idx, grid_points=50):
    """Compute ICE curves for all instances"""
    feature_range = np.linspace(X[:, feature_idx].min(),
                                 X[:, feature_idx].max(),
                                 grid_points)
    
    ice_data = []
    
    for x in X:
        predictions = []
        for value in feature_range:
            x_modified = x.copy()
            x_modified[feature_idx] = value
            pred = model.predict(x_modified.reshape(1, -1))[0]
            predictions.append(pred)
        ice_data.append(predictions)
    
    return feature_range, np.array(ice_data)

# Example
np.random.seed(42)
X_train = np.random.randn(50, 4)

feature_range, ice_curves_data = ice_curves(model, X_train[:10], feature_idx=0)

print("ICE curves computed for 10 instances")
print("Feature range:", f"{feature_range[0]:.2f} to {feature_range[-1]:.2f}")
print("Prediction variation across instances:", 
      f"{ice_curves_data.std(axis=0).mean():.3f}")
```

## Accumulated Local Effects (ALE)

```python
print("\n=== ALE PLOTS ===")
print("""
ALE: Unbiased alternative to PDP

PROBLEM WITH PDP:
  - Creates unrealistic data combinations
  - If features correlated, PDP can be misleading

ALE SOLUTION:
  - Use conditional distribution
  - Compute local effects in small intervals
  - Accumulate effects

Algorithm:
1. Divide feature into intervals
2. In each interval:
   a. Compute average prediction difference
   b. Only use realistic combinations
3. Accumulate effects from first interval

ALE is:
  - Unbiased (no extrapolation)
  - Faster than PDP
  - Better with correlated features
""")

def ale_plot(model, X, feature_idx, n_bins=20):
    """Simplified ALE computation"""
    feature_values = X[:, feature_idx]
    bins = np.percentile(feature_values, np.linspace(0, 100, n_bins + 1))
    
    ale_values = [0]
    
    for i in range(n_bins):
        in_bin = (feature_values >= bins[i]) & (feature_values < bins[i + 1])
        X_in_bin = X[in_bin]
        
        if len(X_in_bin) == 0:
            ale_values.append(ale_values[-1])
            continue
        
        # Predictions at bin boundaries
        X_lower = X_in_bin.copy()
        X_lower[:, feature_idx] = bins[i]
        X_upper = X_in_bin.copy()
        X_upper[:, feature_idx] = bins[i + 1]
        
        # Local effect
        pred_upper = model.predict(X_upper)
        pred_lower = model.predict(X_lower)
        local_effect = np.mean(pred_upper - pred_lower)
        
        ale_values.append(ale_values[-1] + local_effect)
    
    # Center around mean
    ale_values = np.array(ale_values[1:])
    ale_values -= np.mean(ale_values)
    
    return bins[:-1], ale_values

print("ALE provides unbiased feature effect estimates")
```

## Anchors

```python
print("\n=== ANCHORS ===")
print("""
Rule-based explanations with coverage guarantees

Anchor: Set of conditions that "anchor" the prediction
  - If conditions met → same prediction with high probability

Example:
  Prediction: SPAM
  Anchor: IF "free" in text AND "click here" in text
          THEN SPAM (precision: 95%)

Properties:
  - Precision: How often rule holds
  - Coverage: How many instances the rule applies to
  
Trade-off: High precision ↔ Low coverage

Algorithm (beam search):
1. Start with empty anchor
2. Iteratively add conditions
3. Stop when precision threshold met
4. Prefer anchors with higher coverage
""")

def find_anchor(model, x, feature_names, precision_threshold=0.95, 
                n_samples=1000):
    """Simplified anchor finding"""
    n_features = len(x)
    anchor = []
    
    while True:
        best_precision = 0
        best_feature = None
        
        for i in range(n_features):
            if i in [a[0] for a in anchor]:
                continue
            
            # Generate samples matching current anchor + feature i
            samples = np.random.randn(n_samples, n_features)
            
            # Apply anchor conditions
            for feat_idx, value, tolerance in anchor:
                samples[:, feat_idx] = value
            samples[:, i] = x[i]
            
            # Check predictions
            preds = model.predict(samples)
            original_pred = model.predict(x.reshape(1, -1))[0]
            precision = np.mean(preds == original_pred)
            
            if precision > best_precision:
                best_precision = precision
                best_feature = (i, x[i], 0.1)
        
        if best_precision >= precision_threshold:
            anchor.append(best_feature)
            print(f"Found anchor with precision {best_precision:.2f}")
            break
        elif best_feature:
            anchor.append(best_feature)
        else:
            break
    
    return anchor

print("Anchors provide IF-THEN rules with confidence")
```

## Comparing Methods

```python
print("\n=== METHOD COMPARISON ===")
print("""
Method          | Scope    | Output           | Speed    | Best For
----------------|----------|------------------|----------|------------------
Permutation     | Global   | Importance score | Fast     | Overall importance
SHAP            | Both     | Attributions     | Medium   | Detailed analysis
LIME            | Local    | Linear weights   | Fast     | Quick explanations
PDP             | Global   | Effect curves    | Medium   | Feature effects
ICE             | Local    | Instance curves  | Medium   | Heterogeneity
ALE             | Global   | Effect curves    | Fast     | Correlated features
Counterfactuals | Local    | Alternative x    | Varies   | Actionable advice
Anchors         | Local    | Rules            | Slow     | Rule-based systems

RECOMMENDATIONS:
  - Start with permutation importance (global view)
  - Use SHAP for detailed analysis
  - Use counterfactuals for actionable explanations
  - Use ALE when features correlated
""")
```

## Key Points

- **Model-agnostic**: Works with any model via perturbations
- **Counterfactuals**: "What-if" alternative scenarios
- **ICE plots**: Individual-level feature effects
- **ALE**: Unbiased alternative to PDP
- **Anchors**: Rule-based explanations with guarantees
- **Trade-offs**: Choose method based on use case

## Reflection Questions

1. When would counterfactual explanations be more useful than SHAP values?
2. Why might ICE plots reveal information that PDP hides?
3. What makes ALE more suitable than PDP for correlated features?
