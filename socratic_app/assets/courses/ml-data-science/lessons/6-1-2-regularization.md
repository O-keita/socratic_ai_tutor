# Regularization: Ridge, Lasso, and ElasticNet

## Introduction

Regularization adds a penalty term to the loss function, constraining model complexity to prevent overfitting. It's essential when dealing with many features, multicollinearity, or limited data.

## Why Regularization?

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=== WHY REGULARIZATION? ===")
print("""
Problems regularization solves:

1. OVERFITTING
   - Model memorizes training data
   - Poor generalization to new data
   - Especially with many features

2. MULTICOLLINEARITY
   - Correlated features cause unstable coefficients
   - Small data changes → large coefficient changes

3. FEATURE SELECTION
   - Identify important features
   - Reduce model complexity

The Key Idea:
  Instead of: minimize MSE
  Use: minimize MSE + λ × (penalty on coefficients)
  
  λ (lambda/alpha) controls regularization strength:
    - λ = 0: No regularization (standard regression)
    - Large λ: Heavy regularization (simpler model)
""")
```

## Ridge Regression (L2)

```python
print("\n=== RIDGE REGRESSION (L2) ===")
print("""
Loss = MSE + α × Σ(βᵢ²)

Properties:
  - Shrinks coefficients toward zero
  - Never exactly zero (keeps all features)
  - Handles multicollinearity well
  - Closed-form solution (fast)

Use when:
  - Many correlated features
  - All features expected to be relevant
  - Want stable coefficient estimates
""")

# Create data with multicollinearity
n = 100
X1 = np.random.randn(n)
X2 = X1 + np.random.randn(n) * 0.1  # X2 highly correlated with X1
X3 = np.random.randn(n)
X = np.column_stack([X1, X2, X3])
y = 3*X1 + 2*X3 + np.random.randn(n) * 0.5

feature_names = ['X1', 'X2', 'X3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare OLS vs Ridge
print("Correlation between X1 and X2:", np.corrcoef(X1, X2)[0,1].round(3))

ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

print("\nCoefficients (correlated features X1, X2):")
print(f"  OLS:   X1={ols.coef_[0]:.2f}, X2={ols.coef_[1]:.2f}, X3={ols.coef_[2]:.2f}")
print(f"  Ridge: X1={ridge.coef_[0]:.2f}, X2={ridge.coef_[1]:.2f}, X3={ridge.coef_[2]:.2f}")
print("\nNotice Ridge distributes weight between correlated features more evenly.")
```

## Effect of Alpha Parameter

```python
print("\n=== EFFECT OF ALPHA ===")

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
print("Ridge coefficients at different alphas:")
print(f"{'Alpha':>8} | {'X1':>6} | {'X2':>6} | {'X3':>6} | Test R²")
print("-" * 50)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    r2 = ridge.score(X_test_scaled, y_test)
    print(f"{alpha:>8} | {ridge.coef_[0]:>6.2f} | {ridge.coef_[1]:>6.2f} | {ridge.coef_[2]:>6.2f} | {r2:.4f}")

print("""
Observations:
  - Small α: Coefficients similar to OLS
  - Large α: Coefficients shrink toward 0
  - Optimal α: Best bias-variance tradeoff
""")
```

## Lasso Regression (L1)

```python
print("\n=== LASSO REGRESSION (L1) ===")
print("""
Loss = MSE + α × Σ|βᵢ|

Properties:
  - Shrinks coefficients toward zero
  - CAN be exactly zero (automatic feature selection!)
  - Tends to select ONE of correlated features
  - No closed-form solution (iterative)

Use when:
  - Feature selection is important
  - Many features, few expected to be relevant
  - Sparse solution desired
""")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

print("Lasso vs Ridge vs OLS:")
print(f"  OLS:   X1={ols.coef_[0]:.2f}, X2={ols.coef_[1]:.2f}, X3={ols.coef_[2]:.2f}")
print(f"  Ridge: X1={ridge.coef_[0]:.2f}, X2={ridge.coef_[1]:.2f}, X3={ridge.coef_[2]:.2f}")
print(f"  Lasso: X1={lasso.coef_[0]:.2f}, X2={lasso.coef_[1]:.2f}, X3={lasso.coef_[2]:.2f}")

# Feature selection example with more features
print("\n=== LASSO FOR FEATURE SELECTION ===")
np.random.seed(42)
n = 200
n_features = 50
X_large = np.random.randn(n, n_features)
# Only first 5 features are relevant
true_coef = np.zeros(n_features)
true_coef[:5] = [3, -2, 4, 1, -1.5]
y_large = X_large @ true_coef + np.random.randn(n) * 0.5

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

lasso_select = Lasso(alpha=0.1)
lasso_select.fit(X_train_l, y_train_l)

print(f"True relevant features: first 5 out of {n_features}")
print(f"Lasso selected features: {np.sum(lasso_select.coef_ != 0)}")
print(f"Non-zero coefficient indices: {np.where(lasso_select.coef_ != 0)[0][:10]}")
```

## ElasticNet

```python
print("\n=== ELASTICNET ===")
print("""
Loss = MSE + α × (ρ × Σ|βᵢ| + (1-ρ)/2 × Σβᵢ²)

Combines L1 and L2 penalties:
  - l1_ratio (ρ) controls the mix
  - ρ = 0: Pure Ridge
  - ρ = 1: Pure Lasso
  - 0 < ρ < 1: Mix of both

Use when:
  - Want feature selection (like Lasso)
  - But with more stability (like Ridge)
  - Groups of correlated features exist
""")

# Compare on large feature set
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% Lasso, 50% Ridge
enet.fit(X_train_l, y_train_l)

ridge_l = Ridge(alpha=0.1)
ridge_l.fit(X_train_l, y_train_l)

print("Feature selection comparison:")
print(f"  Lasso non-zero: {np.sum(lasso_select.coef_ != 0)}")
print(f"  ElasticNet non-zero: {np.sum(enet.coef_ != 0)}")
print(f"  Ridge non-zero: {np.sum(ridge_l.coef_ != 0)} (Ridge never zeros)")

print("\nTest R² scores:")
print(f"  Lasso: {lasso_select.score(X_test_l, y_test_l):.4f}")
print(f"  ElasticNet: {enet.score(X_test_l, y_test_l):.4f}")
print(f"  Ridge: {ridge_l.score(X_test_l, y_test_l):.4f}")
```

## Cross-Validation for Hyperparameter Tuning

```python
print("\n=== CROSS-VALIDATION FOR ALPHA ===")
print("""
Use cross-validation to find optimal α.
sklearn provides CV versions:
  - RidgeCV
  - LassoCV
  - ElasticNetCV
""")

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# RidgeCV
alphas_ridge = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best Ridge alpha: {ridge_cv.alpha_}")

# LassoCV
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_l, y_train_l)
print(f"Best Lasso alpha: {lasso_cv.alpha_:.4f}")

# ElasticNetCV
enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1], cv=5, random_state=42)
enet_cv.fit(X_train_l, y_train_l)
print(f"Best ElasticNet alpha: {enet_cv.alpha_:.4f}, l1_ratio: {enet_cv.l1_ratio_}")
```

## Choosing Between Methods

```python
print("\n=== CHOOSING THE RIGHT METHOD ===")
print("""
RIDGE (L2):
  ✓ All features expected to be relevant
  ✓ Correlated features exist
  ✓ Want stable coefficient estimates
  ✓ Fast (closed-form solution)
  
LASSO (L1):
  ✓ Feature selection needed
  ✓ Sparse solution expected
  ✓ Want interpretability (fewer features)
  ✓ Some features may be irrelevant

ELASTICNET:
  ✓ Groups of correlated features
  ✓ Want feature selection with stability
  ✓ Hybrid approach needed
  ✓ Large number of features

PRACTICAL ADVICE:
  1. Start with Ridge if unsure
  2. Use Lasso if need feature selection
  3. Try ElasticNet for best of both
  4. Always cross-validate alpha
  5. Scale features before regularization!
""")
```

## Regularization Path

```python
print("\n=== REGULARIZATION PATH ===")
print("""
Plotting coefficients vs alpha shows:
  - How features shrink with more regularization
  - When features drop to zero (Lasso)
  - Which features are most robust
""")

# Compute coefficients across alphas
alphas_path = np.logspace(-3, 3, 50)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas_path:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_l[:, :10], y_train_l)  # First 10 features
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_l[:, :10], y_train_l)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

print("At high alpha (α=100):")
print(f"  Ridge coefs: {ridge_coefs[-10, :5].round(3)}")
print(f"  Lasso coefs: {lasso_coefs[-10, :5].round(3)}")
print("\nLasso drives coefficients to exactly 0, Ridge just shrinks them.")
```

## Key Points

- **Ridge (L2)**: Shrinks coefficients, never to zero, handles multicollinearity
- **Lasso (L1)**: Shrinks coefficients, some to zero, automatic feature selection
- **ElasticNet**: Combines L1 and L2 for balanced approach
- **Alpha (λ)**: Controls regularization strength
- **Cross-validate**: Always use CV to find optimal alpha
- **Scale features**: Required for regularization to work properly

## Reflection Questions

1. Why does Lasso produce sparse solutions while Ridge doesn't?
2. When would ElasticNet be preferred over pure Lasso?
3. How does regularization help with the bias-variance tradeoff?
