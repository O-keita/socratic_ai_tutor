# Linear Regression

## Introduction

Linear regression is the foundation of predictive modeling. It models the relationship between features and a continuous target variable, making it essential for understanding more complex algorithms.

## Core Concepts

### The Linear Model

Linear regression assumes a linear relationship:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

Where:
- $y$ is the target variable
- $\beta_0$ is the intercept (bias)
- $\beta_i$ are the coefficients (weights)
- $x_i$ are the features
- $\epsilon$ is the error term

### Simple Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.2, 5.8, 8.1, 9.9])

# Create and fit model
model = LinearRegression()
model.fit(X, y)

# Get parameters
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# Make predictions
predictions = model.predict(X)
```

### Multiple Linear Regression

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_boston(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
```

### Assumptions of Linear Regression

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

### Loss Function: Mean Squared Error

Linear regression minimizes:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
```

### Interpreting Coefficients

```python
feature_names = ['feature1', 'feature2', 'feature3']
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")
    
# A coefficient of 0.5 means:
# A 1-unit increase in feature → 0.5-unit increase in target
```

---

## Key Points

- Linear regression finds the best-fitting line through data
- Minimizes mean squared error between predictions and actual values
- Coefficients show the relationship strength and direction
- Feature scaling improves numerical stability
- Check assumptions before trusting results

---

## Reflection Questions

1. **Think**: What does a negative coefficient mean? How would you interpret a coefficient of -2.5 for a feature?

2. **Consider**: Why is feature scaling important for interpreting coefficients? What happens if features have very different scales?

3. **Explore**: When might linear regression be inappropriate? What patterns in residuals would suggest violations of assumptions?
