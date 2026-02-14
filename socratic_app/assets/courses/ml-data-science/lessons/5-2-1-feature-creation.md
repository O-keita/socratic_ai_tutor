# Feature Creation and Extraction

## Introduction

Feature creation involves deriving new features from existing data to better capture patterns and improve model performance. Good features often make the difference between mediocre and excellent models.

## Domain-Driven Feature Creation

```python
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

print("=== DOMAIN-DRIVEN FEATURES ===")
print("""
The best features come from domain knowledge!

Questions to ask:
  - What relationships matter in this domain?
  - What patterns do experts look for?
  - What business rules apply?
  - What aggregate statistics are meaningful?
""")

# Example: E-commerce transaction data
df = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'transaction_date': pd.date_range('2023-01-01', periods=9, freq='D'),
    'amount': [100, 50, 200, 150, 80, 30, 40, 50, 100],
    'product_category': ['Electronics', 'Books', 'Electronics', 
                         'Clothing', 'Books', 'Electronics', 
                         'Books', 'Books', 'Clothing']
})
print("Transaction Data:")
print(df)

# Create user-level features
print("\n=== USER AGGREGATE FEATURES ===")
user_features = df.groupby('user_id').agg({
    'amount': ['sum', 'mean', 'std', 'count', 'max'],
    'product_category': 'nunique'
}).round(2)
user_features.columns = ['total_spent', 'avg_transaction', 'std_transaction', 
                         'transaction_count', 'max_transaction', 'unique_categories']
user_features['avg_transactions_per_category'] = (
    user_features['transaction_count'] / user_features['unique_categories']
).round(2)
print(user_features)
```

## Date and Time Features

```python
print("\n=== DATE/TIME FEATURES ===")
print("""
Datetime columns contain rich information:
  - Cyclical patterns (day of week, month, hour)
  - Trends over time
  - Special periods (holidays, weekends)
""")

df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
    'value': np.random.randn(100) * 10 + 50
})

# Extract datetime components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour'] = df['timestamp'].dt.hour
df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_morning'] = df['hour'].between(6, 11).astype(int)
df['quarter'] = df['timestamp'].dt.quarter

print("Datetime features:")
print(df.head(10))

# Cyclical encoding for cyclical features
print("\n=== CYCLICAL ENCODING ===")
print("""
For cyclical features (hour, day of week, month),
use sine/cosine transformation:

  sin(2π * x / max_x)
  cos(2π * x / max_x)

This makes 23:00 close to 00:00 in feature space!
""")

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print("Cyclical encoding (sample):")
print(df[['hour', 'hour_sin', 'hour_cos', 'day_of_week', 'dow_sin', 'dow_cos']].head(10).round(3))
```

## Polynomial and Interaction Features

```python
print("\n=== POLYNOMIAL FEATURES ===")
print("""
Create polynomial combinations:
  - x² (quadratic term)
  - x × y (interaction)
  - x² × y, etc.

Useful for capturing non-linear relationships
in linear models.
""")

from sklearn.preprocessing import PolynomialFeatures

# Simple example
X = np.array([[1, 2], [3, 4], [5, 6]])
print("Original features:")
print(X)

# Degree 2 polynomial
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"\nPolynomial features (degree=2):")
print(f"Feature names: {poly.get_feature_names_out(['x1', 'x2'])}")
print(X_poly)

# Interaction only
print("\n=== INTERACTION FEATURES ONLY ===")
poly_interact = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interact = poly_interact.fit_transform(X)
print(f"Feature names: {poly_interact.get_feature_names_out(['x1', 'x2'])}")
print(X_interact)
```

## Binning and Discretization

```python
print("\n=== BINNING / DISCRETIZATION ===")
print("""
Convert continuous to categorical:
  - Equal-width bins
  - Equal-frequency bins (quantiles)
  - Custom bins (domain-based)

Why bin?
  - Capture non-linear relationships
  - Reduce noise
  - Handle outliers
  - Meet business requirements
""")

data = np.concatenate([
    np.random.normal(30, 5, 100),
    np.random.normal(60, 10, 100)
])
df_bin = pd.DataFrame({'age': data})

# Equal-width bins
df_bin['age_bins_width'] = pd.cut(df_bin['age'], bins=5)
print("Equal-width bins:")
print(df_bin['age_bins_width'].value_counts().sort_index())

# Equal-frequency bins (quantiles)
df_bin['age_bins_quantile'] = pd.qcut(df_bin['age'], q=5)
print("\nEqual-frequency bins (quantiles):")
print(df_bin['age_bins_quantile'].value_counts().sort_index())

# Custom bins
custom_bins = [0, 18, 35, 50, 65, 100]
custom_labels = ['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
df_bin['age_group'] = pd.cut(df_bin['age'], bins=custom_bins, labels=custom_labels)
print("\nCustom bins:")
print(df_bin['age_group'].value_counts())
```

## Text Features

```python
print("\n=== TEXT FEATURES ===")
print("""
Basic text features:
  - Length (characters, words)
  - Word count
  - Average word length
  - Special character count
  - Uppercase ratio
""")

texts = pd.Series([
    "Hello World!",
    "This is a longer text with more words.",
    "SHORT",
    "Machine Learning is AMAZING!!!",
    "Data Science 101: Introduction to Python"
])

df_text = pd.DataFrame({'text': texts})
df_text['char_count'] = df_text['text'].str.len()
df_text['word_count'] = df_text['text'].str.split().str.len()
df_text['avg_word_length'] = df_text['text'].apply(
    lambda x: np.mean([len(w) for w in x.split()])
).round(2)
df_text['digit_count'] = df_text['text'].str.count(r'\d')
df_text['special_count'] = df_text['text'].str.count(r'[!@#$%^&*()]')
df_text['uppercase_ratio'] = df_text['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x)
).round(2)

print("Text features:")
print(df_text)
```

## Aggregation Features

```python
print("\n=== AGGREGATION FEATURES ===")
print("""
For hierarchical data, aggregate child records to parent level:
  - Counts
  - Sums, means, medians
  - Min, max, range
  - Standard deviation
  - First, last, most recent
""")

# Transactions per customer
transactions = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
    'amount': [100, 200, 150, 80, 120, 50, 60, 70, 80, 90],
    'days_ago': [1, 3, 7, 2, 10, 1, 5, 10, 15, 30]
})

print("Transaction data:")
print(transactions)

# Aggregate to customer level
customer_features = transactions.groupby('customer_id').agg({
    'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
    'days_ago': ['min', 'max', 'mean']
})

# Flatten column names
customer_features.columns = ['_'.join(col) for col in customer_features.columns]

# Additional derived features
customer_features['amount_range'] = (
    customer_features['amount_max'] - customer_features['amount_min']
)
customer_features['recency'] = customer_features['days_ago_min']
customer_features['tenure'] = customer_features['days_ago_max']

print("\nCustomer-level features:")
print(customer_features.round(2))
```

## Ratio and Comparison Features

```python
print("\n=== RATIO FEATURES ===")
print("""
Ratios capture relative relationships:
  - Value / Total
  - Current / Previous
  - Current / Average
  - Part / Whole
""")

df = pd.DataFrame({
    'revenue': [1000, 1500, 800, 2000],
    'cost': [600, 900, 400, 1100],
    'employees': [10, 15, 8, 20],
    'customers': [100, 120, 80, 200]
})

# Create ratio features
df['profit'] = df['revenue'] - df['cost']
df['profit_margin'] = (df['profit'] / df['revenue']).round(3)
df['cost_ratio'] = (df['cost'] / df['revenue']).round(3)
df['revenue_per_employee'] = (df['revenue'] / df['employees']).round(2)
df['revenue_per_customer'] = (df['revenue'] / df['customers']).round(2)
df['customers_per_employee'] = (df['customers'] / df['employees']).round(2)

print("With ratio features:")
print(df)
```

## Key Points

- **Domain knowledge**: Best features come from understanding the problem
- **Datetime**: Extract components, use cyclical encoding
- **Polynomials**: Capture non-linear relationships
- **Binning**: Convert continuous to categorical
- **Text features**: Length, counts, ratios
- **Aggregation**: Summarize hierarchical data
- **Ratios**: Capture relative relationships

## Reflection Questions

1. How do you decide which feature transformations to try?
2. When might binning improve model performance?
3. How does cyclical encoding help with periodic features?
