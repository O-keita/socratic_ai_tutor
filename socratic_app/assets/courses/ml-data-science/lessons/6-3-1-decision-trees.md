# Decision Trees

## Introduction

Decision trees are intuitive, interpretable models that make predictions by learning simple decision rules from data. They form the foundation for powerful ensemble methods.

## Core Concepts

### How Decision Trees Work

A decision tree splits data based on feature values:

```
                [Age <= 30?]
                /          \
              Yes           No
              /              \
        [Income > 50k?]    [Married?]
        /          \       /        \
      Yes          No    Yes        No
       ↓            ↓      ↓          ↓
    Approve      Deny   Approve    Deny
```

### Building a Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Classification
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
clf.fit(X_train, y_train)

# Regression
reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train, y_train)
```

### Splitting Criteria

**For Classification:**
- **Gini Impurity**: Measures probability of incorrect classification
  $$Gini = 1 - \sum_{i=1}^{c} p_i^2$$
  
- **Entropy**: Measures information gain
  $$Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

**For Regression:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### Controlling Tree Complexity

```python
# Key hyperparameters
tree = DecisionTreeClassifier(
    max_depth=10,           # Maximum tree depth
    min_samples_split=20,   # Min samples to split node
    min_samples_leaf=5,     # Min samples in leaf
    max_features='sqrt',    # Features to consider per split
    max_leaf_nodes=50       # Maximum leaf nodes
)
```

### Visualizing Decision Trees

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True)
plt.show()
```

### Feature Importance

```python
# Get feature importances
importances = clf.feature_importances_

# Sort and display
indices = np.argsort(importances)[::-1]
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
```

### Advantages and Disadvantages

**Advantages:**
- Easy to understand and visualize
- Handles both numerical and categorical data
- Requires little data preprocessing
- Feature importance built-in

**Disadvantages:**
- Prone to overfitting
- Sensitive to small data changes
- Can create biased trees with imbalanced data
- Not smooth predictions (step-wise)

---

## Key Points

- Trees make decisions through sequential splits
- Gini impurity and entropy measure split quality
- Limit depth and require minimum samples to prevent overfitting
- Feature importance shows which features drive predictions
- Single trees are interpretable but often overfit

---

## Reflection Questions

1. **Think**: Why does limiting tree depth help prevent overfitting? What information might be lost with a shallow tree?

2. **Consider**: How does a decision tree handle missing values? What strategies could you use to address this?

3. **Explore**: Why might feature importance in a decision tree differ from correlation with the target? What does importance actually measure?
