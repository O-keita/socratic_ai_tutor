# Data Transformations with Pandas

## Introduction

Data transformation is the process of converting data from one format or structure to another. Pandas provides powerful methods for reshaping, aggregating, and manipulating DataFrames to prepare data for analysis and modeling.

## Creating and Modifying Columns

```python
import pandas as pd
import numpy as np

np.random.seed(42)

# Sample data
df = pd.DataFrame({
    'product': ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry'],
    'price': [1.20, 0.50, 2.50, 3.00, 4.50],
    'quantity': [100, 150, 75, 50, 25],
    'category': ['Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit']
})

print("=== CREATING COLUMNS ===")
print(f"Original:\n{df}\n")

# Create new column from calculation
df['total_value'] = df['price'] * df['quantity']
print(f"Added total_value:\n{df}\n")

# Create column with conditional logic
df['price_tier'] = np.where(df['price'] > 2, 'Premium', 'Standard')
print(f"Added price_tier:\n{df}\n")

# Multiple conditions with np.select
conditions = [
    df['price'] < 1,
    df['price'] < 3,
    df['price'] >= 3
]
choices = ['Budget', 'Mid-range', 'Premium']
df['price_category'] = np.select(conditions, choices)
print(f"Added price_category:\n{df}")
```

## Apply and Map Functions

```python
print("\n=== APPLY AND MAP ===")

# apply() - apply function to each element, row, or column
df['price_formatted'] = df['price'].apply(lambda x: f"${x:.2f}")
print(f"Formatted prices:\n{df[['product', 'price', 'price_formatted']]}\n")

# Apply to entire row
def row_summary(row):
    return f"{row['product']}: {row['quantity']} units @ ${row['price']}"

df['summary'] = df.apply(row_summary, axis=1)
print(f"Row summaries:\n{df[['summary']]}\n")

# map() - map values using dictionary or function
price_map = {1.20: 'Low', 0.50: 'Very Low', 2.50: 'Medium', 3.00: 'High', 4.50: 'Very High'}
df['price_label'] = df['price'].map(price_map)
print(f"Mapped prices:\n{df[['product', 'price', 'price_label']]}")
```

## String Transformations

```python
print("\n=== STRING TRANSFORMATIONS ===")

df_text = pd.DataFrame({
    'name': ['  John Smith  ', 'JANE DOE', 'bob wilson', 'Alice Johnson'],
    'email': ['john@email.com', 'jane@EMAIL.com', 'BOB@email.COM', 'alice@Email.com']
})
print(f"Original:\n{df_text}\n")

# String methods via .str accessor
df_text['name_clean'] = df_text['name'].str.strip()
df_text['name_upper'] = df_text['name'].str.upper()
df_text['name_lower'] = df_text['name'].str.lower()
df_text['name_title'] = df_text['name'].str.strip().str.title()

print(f"String transformations:\n{df_text[['name', 'name_clean', 'name_title']]}\n")

# Extract parts of strings
df_text['email_domain'] = df_text['email'].str.split('@').str[1].str.lower()
df_text['first_name'] = df_text['name'].str.strip().str.split().str[0].str.title()
print(f"Extracted parts:\n{df_text[['name', 'first_name', 'email', 'email_domain']]}")
```

## Date Transformations

```python
print("\n=== DATE TRANSFORMATIONS ===")

df_dates = pd.DataFrame({
    'date_str': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-12-25'],
    'value': [100, 150, 200, 250]
})

# Convert to datetime
df_dates['date'] = pd.to_datetime(df_dates['date_str'])

# Extract date components
df_dates['year'] = df_dates['date'].dt.year
df_dates['month'] = df_dates['date'].dt.month
df_dates['month_name'] = df_dates['date'].dt.month_name()
df_dates['day'] = df_dates['date'].dt.day
df_dates['day_name'] = df_dates['date'].dt.day_name()
df_dates['quarter'] = df_dates['date'].dt.quarter
df_dates['week'] = df_dates['date'].dt.isocalendar().week

print(f"Date components:\n{df_dates}")
```

## Binning and Discretization

```python
print("\n=== BINNING (DISCRETIZATION) ===")

df_scores = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'score': [92, 78, 85, 67, 95]
})
print(f"Original scores:\n{df_scores}\n")

# pd.cut() - bin into equal-width bins
df_scores['score_bin'] = pd.cut(df_scores['score'], bins=3, labels=['Low', 'Medium', 'High'])
print(f"Equal-width bins:\n{df_scores}\n")

# Custom bin edges
bins = [0, 60, 70, 80, 90, 100]
labels = ['F', 'D', 'C', 'B', 'A']
df_scores['grade'] = pd.cut(df_scores['score'], bins=bins, labels=labels)
print(f"Custom grade bins:\n{df_scores}\n")

# pd.qcut() - bin into equal-frequency bins (quantiles)
df_scores['quartile'] = pd.qcut(df_scores['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(f"Quantile bins:\n{df_scores}")
```

## Reshaping: Pivot and Melt

```python
print("\n=== RESHAPING DATA ===")

# Sample sales data
sales = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})
print(f"Original (long format):\n{sales}\n")

# Pivot: long to wide
pivoted = sales.pivot(index='date', columns='product', values='sales')
print(f"Pivoted (wide format):\n{pivoted}\n")

# Melt: wide to long
melted = pivoted.reset_index().melt(id_vars='date', var_name='product', value_name='sales')
print(f"Melted (back to long):\n{melted}")
```

## Replace Values

```python
print("\n=== REPLACING VALUES ===")

df_replace = pd.DataFrame({
    'status': ['active', 'inactive', 'pending', 'active', 'unknown'],
    'code': [1, 2, 3, 1, -1]
})
print(f"Original:\n{df_replace}\n")

# Replace single value
df_replace['status_clean'] = df_replace['status'].replace('unknown', 'pending')
print(f"Replace single value:\n{df_replace}\n")

# Replace multiple values with dictionary
status_map = {'active': 'A', 'inactive': 'I', 'pending': 'P', 'unknown': 'U'}
df_replace['status_code'] = df_replace['status'].replace(status_map)
print(f"Replace with mapping:\n{df_replace}\n")

# Replace in entire DataFrame
df_replace_all = df_replace.replace({-1: np.nan, 'unknown': np.nan})
print(f"Replace -1 and 'unknown' with NaN:\n{df_replace_all}")
```

## Type Conversions

```python
print("\n=== TYPE CONVERSIONS ===")

df_types = pd.DataFrame({
    'id': ['1', '2', '3', '4'],
    'value': ['10.5', '20.3', '15.7', '25.1'],
    'flag': ['True', 'False', 'True', 'False']
})
print(f"Original types:\n{df_types.dtypes}\n")

# Convert types
df_types['id'] = df_types['id'].astype(int)
df_types['value'] = df_types['value'].astype(float)
df_types['flag'] = df_types['flag'].map({'True': True, 'False': False})

print(f"Converted types:\n{df_types.dtypes}\n")
print(f"Data:\n{df_types}")
```

## Key Points

- **New columns**: Create with calculations or conditional logic
- **apply()**: Apply custom functions to rows/columns
- **map()**: Transform values using dictionary or function
- **String methods**: Use `.str` accessor for text operations
- **Date methods**: Use `.dt` accessor for datetime operations
- **Binning**: `pd.cut()` for equal-width, `pd.qcut()` for quantiles
- **Reshape**: `pivot()` for long-to-wide, `melt()` for wide-to-long
- **replace()**: Substitute values in columns or entire DataFrame

## Reflection Questions

1. When would you use `apply()` vs `map()` for transformations?
2. Why might you convert a continuous variable to categories using binning?
3. What's the difference between `pd.cut()` and `pd.qcut()`?
