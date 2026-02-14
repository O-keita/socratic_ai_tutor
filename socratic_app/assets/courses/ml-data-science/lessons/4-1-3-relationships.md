# Exploring Relationships Between Variables

## Introduction

Understanding relationships between variables is crucial for feature selection, hypothesis generation, and model building. This lesson covers techniques for exploring associations in data.

## Correlation Analysis

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

print("=== CORRELATION ANALYSIS ===")
print("""
Correlation measures the LINEAR relationship between two variables.

Types of Correlation:
  1. Pearson (r): Linear relationship, continuous data
  2. Spearman (ρ): Monotonic relationship, ordinal data
  3. Kendall (τ): Concordance of pairs, robust to outliers
""")

# Create sample data
n = 200
x = np.random.uniform(0, 100, n)
y_linear = 2*x + np.random.normal(0, 20, n)
y_nonlinear = x**2 + np.random.normal(0, 500, n)
y_none = np.random.uniform(0, 100, n)

# Calculate correlations
print("\nPearson Correlations:")
print(f"  Linear relationship: r = {np.corrcoef(x, y_linear)[0,1]:.3f}")
print(f"  Quadratic relationship: r = {np.corrcoef(x, y_nonlinear)[0,1]:.3f}")
print(f"  No relationship: r = {np.corrcoef(x, y_none)[0,1]:.3f}")

print("\nSpearman Correlations:")
print(f"  Linear relationship: ρ = {stats.spearmanr(x, y_linear)[0]:.3f}")
print(f"  Quadratic relationship: ρ = {stats.spearmanr(x, y_nonlinear)[0]:.3f}")
print(f"  No relationship: ρ = {stats.spearmanr(x, y_none)[0]:.3f}")
```

## Correlation Matrix

```python
print("\n=== CORRELATION MATRIX ===")

# Create a DataFrame with multiple variables
df = pd.DataFrame({
    'age': np.random.normal(40, 12, 200),
    'income': np.random.exponential(50000, 200),
    'education_years': np.random.normal(14, 3, 200),
    'experience': np.random.normal(15, 8, 200)
})
# Add some correlations
df['income'] = df['income'] + 2000*df['education_years'] + 1500*df['experience']
df['experience'] = df['age'] - 18 - df['education_years'] + np.random.normal(0, 2, 200)

print("Correlation Matrix:")
corr_matrix = df.corr()
print(corr_matrix.round(3))

print("\nCorrelation Strength Guide:")
print("""
  |r| < 0.1: Negligible
  0.1 ≤ |r| < 0.3: Weak
  0.3 ≤ |r| < 0.5: Moderate
  0.5 ≤ |r| < 0.7: Strong
  |r| ≥ 0.7: Very strong
""")

# Find strongest correlations
print("Strongest correlations (excluding self):")
# Get upper triangle
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Stack and sort
correlations = upper_tri.stack().sort_values(ascending=False)
print(correlations)
```

## Scatter Plot Analysis

```python
print("\n=== SCATTER PLOT ANALYSIS ===")
print("""
What to look for in scatter plots:

1. DIRECTION
   - Positive: Both variables increase together
   - Negative: One increases, other decreases
   - None: No clear pattern

2. FORM
   - Linear: Points follow a straight line
   - Curved: Non-linear relationship
   - Clusters: Distinct groups

3. STRENGTH
   - Strong: Points close to pattern
   - Weak: Points scattered around pattern

4. OUTLIERS
   - Points far from the general pattern
   - May indicate errors or interesting cases
""")

# Identify potential patterns
print("Checking relationships numerically:")
for col1 in ['age', 'education_years']:
    for col2 in ['income', 'experience']:
        r, p = stats.pearsonr(df[col1], df[col2])
        print(f"  {col1} vs {col2}: r={r:.3f}, p={p:.4f}")
```

## Categorical vs Numeric Relationships

```python
print("\n=== CATEGORICAL VS NUMERIC ===")

# Create data with groups
df['education_level'] = pd.cut(df['education_years'], 
                               bins=[0, 12, 16, 25],
                               labels=['High School', 'Bachelors', 'Graduate'])

print("Income by Education Level:")
print(df.groupby('education_level')['income'].agg(['mean', 'median', 'std', 'count']))

# Effect size - Cohen's d
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

high_school = df[df['education_level'] == 'High School']['income']
graduate = df[df['education_level'] == 'Graduate']['income']
d = cohens_d(graduate, high_school)

print(f"\nEffect Size (Graduate vs High School):")
print(f"  Cohen's d = {d:.3f}")
print("""
  d interpretation:
    0.2: Small effect
    0.5: Medium effect
    0.8: Large effect
""")
```

## Categorical vs Categorical Relationships

```python
print("\n=== CATEGORICAL VS CATEGORICAL ===")

# Create categorical variables
df['income_level'] = pd.qcut(df['income'], q=3, labels=['Low', 'Medium', 'High'])

print("Cross-tabulation:")
ct = pd.crosstab(df['education_level'], df['income_level'])
print(ct)

# Chi-square test for independence
chi2, p_value, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square test for independence:")
print(f"  Chi-square statistic: {chi2:.2f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Degrees of freedom: {dof}")

# Cramer's V (effect size for chi-square)
n = ct.sum().sum()
min_dim = min(ct.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"  Cramer's V: {cramers_v:.3f}")
print("""
  Cramer's V interpretation:
    0.1: Small association
    0.3: Medium association
    0.5: Large association
""")

# Proportions
print("\nRow proportions (education → income level):")
print(ct.div(ct.sum(axis=1), axis=0).round(3))
```

## Multi-Variable Relationships

```python
print("\n=== MULTI-VARIABLE RELATIONSHIPS ===")

# Partial correlation
print("""
Partial Correlation:
  Correlation between X and Y, controlling for Z
  Removes the effect of Z from both variables
""")

# Simple example
from scipy import stats

def partial_correlation(x, y, control):
    """Calculate partial correlation between x and y, controlling for control."""
    # Residuals of x on control
    slope_xc, intercept_xc, _, _, _ = stats.linregress(control, x)
    residuals_x = x - (slope_xc * control + intercept_xc)
    
    # Residuals of y on control
    slope_yc, intercept_yc, _, _, _ = stats.linregress(control, y)
    residuals_y = y - (slope_yc * control + intercept_yc)
    
    # Correlation of residuals
    return np.corrcoef(residuals_x, residuals_y)[0, 1]

# Example
age = df['age'].values
income = df['income'].values
experience = df['experience'].values

r_age_income = np.corrcoef(age, income)[0, 1]
partial_r = partial_correlation(age, income, experience)

print(f"Correlation (age, income): {r_age_income:.3f}")
print(f"Partial correlation (age, income | experience): {partial_r:.3f}")
print("The relationship changes when controlling for experience!")
```

## Detecting Non-Linear Relationships

```python
print("\n=== NON-LINEAR RELATIONSHIPS ===")

# Generate non-linear data
x = np.linspace(0, 10, 100)
y_quad = x**2 + np.random.normal(0, 5, 100)
y_exp = np.exp(0.3*x) + np.random.normal(0, 2, 100)
y_log = 20*np.log(x + 1) + np.random.normal(0, 2, 100)

print("Detecting non-linearity:")
print("\nPearson vs Spearman comparison:")
datasets = [('Quadratic', y_quad), ('Exponential', y_exp), ('Logarithmic', y_log)]

for name, y in datasets:
    pearson = np.corrcoef(x, y)[0, 1]
    spearman = stats.spearmanr(x, y)[0]
    print(f"  {name}:")
    print(f"    Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
    print(f"    Difference: {abs(spearman - pearson):.3f}")

print("""
Key insight:
  - If Spearman >> Pearson: Monotonic but non-linear
  - Spearman captures rank order, not linearity
  - Large difference suggests transformation needed
""")
```

## Key Points

- **Pearson r**: Measures linear relationship strength
- **Spearman ρ**: Measures monotonic relationship (robust)
- **Correlation matrix**: Overview of all pairwise relationships
- **Chi-square**: Tests association between categorical variables
- **Cramer's V**: Effect size for categorical associations
- **Partial correlation**: Controls for confounding variables
- **Spearman vs Pearson**: Large difference suggests non-linearity

## Reflection Questions

1. Why might Spearman correlation be higher than Pearson for the same data?
2. How would you investigate a moderate correlation to understand the underlying relationship?
3. What does a strong correlation tell you about causation?
