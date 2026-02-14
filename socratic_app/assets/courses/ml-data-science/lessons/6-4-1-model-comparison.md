# Model Comparison and Selection

## Introduction

Choosing the right model for your problem requires systematic comparison across multiple metrics, understanding each model's strengths and weaknesses, and considering practical constraints.

## A Systematic Approach

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
import time

# Models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

np.random.seed(42)

print("=== MODEL COMPARISON FRAMEWORK ===")
print("""
When comparing models, consider:

1. PERFORMANCE METRICS
   - Accuracy, F1, AUC-ROC, etc.
   - Choose metrics relevant to problem

2. TRAINING/PREDICTION TIME
   - How long to train?
   - How fast are predictions?

3. INTERPRETABILITY
   - Can you explain predictions?
   - Regulatory requirements?

4. SCALABILITY
   - Dataset size constraints?
   - Memory requirements?

5. HYPERPARAMETER SENSITIVITY
   - How much tuning needed?
   - Stability across different runs?
""")
```

## Comparing Multiple Models

```python
print("\n=== COMPARING MODELS ===")

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, n_clusters_per_class=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True)
}

# Scale data (needed for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
results = []
for name, model in models.items():
    # Use scaled data for distance-based models
    if name in ['KNN', 'SVM (RBF)', 'Logistic Regression']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Time training
    start = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - start
    
    # Time prediction
    start = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - start
    
    # Cross-validation score
    if name in ['KNN', 'SVM (RBF)', 'Logistic Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Test accuracy
    test_acc = model.score(X_te, y_test)
    
    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Acc': test_acc,
        'Train Time (s)': train_time,
        'Pred Time (ms)': pred_time * 1000
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

## Statistical Significance

```python
print("\n=== STATISTICAL SIGNIFICANCE ===")
print("""
When comparing models, check if differences are significant:

1. CROSS-VALIDATION VARIABILITY
   - Compare CV means AND standard deviations
   - Overlapping error bars = may not be different

2. PAIRED TESTS
   - Compare models on same folds
   - Use paired t-test or Wilcoxon test

3. MULTIPLE COMPARISONS
   - When comparing many models
   - Adjust for multiple testing (Bonferroni)
""")

from sklearn.model_selection import cross_val_score, RepeatedKFold
from scipy import stats

# Get detailed CV scores
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

scores_lr = cross_val_score(LogisticRegression(max_iter=1000), 
                            X_train_scaled, y_train, cv=cv)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), 
                            X_train, y_train, cv=cv)

print(f"Logistic Regression: {scores_lr.mean():.3f} ± {scores_lr.std():.3f}")
print(f"Random Forest: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")

# Paired t-test
t_stat, p_value = stats.ttest_rel(scores_rf, scores_lr)
print(f"\nPaired t-test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  Difference is statistically significant")
else:
    print("  Difference is NOT statistically significant")
```

## Model Selection Criteria

```python
print("\n=== MODEL SELECTION CRITERIA ===")
print("""
OCCAM'S RAZOR:
  "Among models that perform similarly, prefer the simpler one"

Consider trade-offs:

              | Accuracy | Interpretability | Speed | Scaling
--------------+----------+------------------+-------+--------
Lin. Reg.     |   Low    |      High        | Fast  |  Good
Log. Reg.     |   Med    |      High        | Fast  |  Good
KNN           |   Med    |      Med         | Slow  |  Poor
Dec. Tree     |   Med    |      High        | Fast  |  Good
Rand. Forest  |   High   |      Low         | Med   |  Good
Grad. Boost   |   High   |      Low         | Slow  |  Med
SVM           |   High   |      Low         | Med   |  Poor
Neural Net    |   High   |      Low         | Slow  |  Good
""")

print("""
PRACTICAL RECOMMENDATIONS:

Start Simple:
  1. Baseline with simple model (LogReg, DecTree)
  2. If not sufficient, try ensembles
  3. Only use complex models if needed

For Tabular Data (Structured):
  → Gradient Boosting (XGBoost, LightGBM)
  → Random Forest as backup

For High Interpretability Needs:
  → Logistic/Linear Regression
  → Decision Trees
  → Rule-based systems

For Quick Prototyping:
  → LogisticRegression (classification)
  → Ridge/Lasso (regression)
  → Random Forest

For Maximum Accuracy:
  → Ensemble of multiple models
  → Hyperparameter-tuned Gradient Boosting
""")
```

## Using Pipelines for Fair Comparison

```python
print("\n=== PIPELINES FOR FAIR COMPARISON ===")

from sklearn.pipeline import Pipeline

# Proper comparison with pipelines
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Random Forest': Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
}

print("Cross-validation with proper pipelines:")
for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")

print("""
Why Pipelines?
  - Proper train/test separation
  - Scaler fit on training data only
  - Prevents data leakage
  - Reproducible workflow
""")
```

## The Final Model Selection Process

```python
print("\n=== FINAL MODEL SELECTION PROCESS ===")
print("""
STEP 1: Define Success Criteria
  - Primary metric (what to optimize)
  - Secondary metrics (constraints)
  - Acceptable thresholds

STEP 2: Establish Baseline
  - Simple model (LogReg, mean predictor)
  - Sets minimum bar to beat

STEP 3: Train Multiple Candidates
  - Different model families
  - Cross-validation for each

STEP 4: Compare Models
  - Statistical tests
  - Consider variance
  - Check for overfitting

STEP 5: Tune Top Candidates
  - Grid/Random search
  - More detailed comparison

STEP 6: Final Evaluation
  - Held-out test set (used ONCE)
  - Report final performance

STEP 7: Document Decision
  - Why this model?
  - Trade-offs considered
  - Deployment considerations
""")
```

## Model Deployment Considerations

```python
print("\n=== DEPLOYMENT CONSIDERATIONS ===")
print("""
BEFORE CHOOSING FINAL MODEL:

1. INFERENCE LATENCY
   - How fast must predictions be?
   - Real-time vs batch?

2. MODEL SIZE
   - Storage requirements?
   - Mobile/edge deployment?

3. UPDATE FREQUENCY
   - How often will model retrain?
   - Concept drift concerns?

4. MONITORING NEEDS
   - How to detect degradation?
   - Feedback loop availability?

5. REGULATORY/COMPLIANCE
   - Explainability requirements?
   - Audit trails needed?

6. MAINTENANCE BURDEN
   - Team expertise?
   - Infrastructure requirements?

Sometimes a simpler model with 1% less accuracy
is the RIGHT choice for production!
""")
```

## Key Points

- **Compare systematically**: Same data, same CV, same metrics
- **Statistical testing**: Check if differences are significant
- **Consider trade-offs**: Accuracy vs speed vs interpretability
- **Use pipelines**: Ensure fair comparison without data leakage
- **Start simple**: Complex models only if needed
- **Deployment matters**: Production constraints affect choice
- **Document decisions**: Record why you chose the model

## Reflection Questions

1. When might you prefer a model with 5% lower accuracy?
2. How do you handle situations where different metrics favor different models?
3. What factors beyond accuracy should influence model selection in a business context?
