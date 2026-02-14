# GroupBy Operations in Pandas

## Introduction

GroupBy operations are fundamental for data analysis, allowing you to split data into groups, apply functions to each group, and combine results. This "split-apply-combine" pattern is essential for aggregating and summarizing data.

## Basic GroupBy

```python
import pandas as pd
import numpy as np

np.random.seed(42)

# Sample sales data
sales = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
    'product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'sales': [100, 150, 200, 175, 125, 140, 180, 160],
    'quantity': [10, 15, 20, 18, 12, 14, 18, 16]
})

print("=== BASIC GROUPBY ===")
print(f"Data:\n{sales}\n")

# Group by single column
grouped = sales.groupby('region')
print(f"GroupBy object: {grouped}")
print(f"Groups: {grouped.groups.keys()}\n")

# Aggregation: sum
region_totals = sales.groupby('region')['sales'].sum()
print(f"Sales by region (sum):\n{region_totals}\n")

# Multiple aggregations
region_stats = sales.groupby('region')['sales'].agg(['sum', 'mean', 'count'])
print(f"Sales statistics by region:\n{region_stats}")
```

## Common Aggregation Functions

```python
print("\n=== AGGREGATION FUNCTIONS ===")

# Built-in aggregations
print("Built-in aggregation methods:")
print(f"  sum():   {sales.groupby('region')['sales'].sum().tolist()}")
print(f"  mean():  {sales.groupby('region')['sales'].mean().tolist()}")
print(f"  median(): {sales.groupby('region')['sales'].median().tolist()}")
print(f"  min():   {sales.groupby('region')['sales'].min().tolist()}")
print(f"  max():   {sales.groupby('region')['sales'].max().tolist()}")
print(f"  count(): {sales.groupby('region')['sales'].count().tolist()}")
print(f"  std():   {sales.groupby('region')['sales'].std().round(2).tolist()}")

# Describe all at once
print(f"\nFull description:\n{sales.groupby('region')['sales'].describe()}")
```

## Multiple Columns Aggregation

```python
print("\n=== AGGREGATE MULTIPLE COLUMNS ===")

# Aggregate all numeric columns
all_stats = sales.groupby('region').sum()
print(f"Sum all numeric columns:\n{all_stats}\n")

# Select specific columns
selected = sales.groupby('region')[['sales', 'quantity']].mean()
print(f"Mean of selected columns:\n{selected}\n")

# Different aggregations for different columns
custom_agg = sales.groupby('region').agg({
    'sales': 'sum',
    'quantity': 'mean'
})
print(f"Custom aggregations:\n{custom_agg}")
```

## Named Aggregations

```python
print("\n=== NAMED AGGREGATIONS ===")

# Use named aggregation for clear column names
result = sales.groupby('region').agg(
    total_sales=('sales', 'sum'),
    avg_sales=('sales', 'mean'),
    total_qty=('quantity', 'sum'),
    num_transactions=('sales', 'count')
)
print(f"Named aggregations:\n{result}")
```

## Multiple Grouping Columns

```python
print("\n=== GROUP BY MULTIPLE COLUMNS ===")

# Group by region AND product
multi_group = sales.groupby(['region', 'product'])['sales'].sum()
print(f"Sales by region and product:\n{multi_group}\n")

# Unstack to create pivot-like table
unstacked = multi_group.unstack()
print(f"Unstacked (pivot view):\n{unstacked}\n")

# Reset index to get back DataFrame
reset = sales.groupby(['region', 'product'])['sales'].sum().reset_index()
print(f"Reset index:\n{reset}")
```

## Custom Aggregation Functions

```python
print("\n=== CUSTOM AGGREGATION ===")

# Define custom function
def range_calc(x):
    return x.max() - x.min()

def coefficient_of_variation(x):
    return x.std() / x.mean() if x.mean() != 0 else 0

# Apply custom function
sales_range = sales.groupby('region')['sales'].agg(range_calc)
print(f"Sales range by region:\n{sales_range}\n")

# Multiple custom functions
custom_stats = sales.groupby('region')['sales'].agg([
    ('range', range_calc),
    ('cv', coefficient_of_variation),
    ('total', 'sum')
])
print(f"Custom statistics:\n{custom_stats}")
```

## Transform: Broadcast Results Back

```python
print("\n=== TRANSFORM ===")

# transform() returns same-sized output
# Useful for creating group-level features

sales_copy = sales.copy()

# Add group mean as new column
sales_copy['region_avg'] = sales.groupby('region')['sales'].transform('mean')
print(f"Data with group mean:\n{sales_copy}\n")

# Normalize within groups
sales_copy['sales_normalized'] = sales.groupby('region')['sales'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(f"With normalized sales:\n{sales_copy[['region', 'sales', 'sales_normalized']]}\n")

# Rank within groups
sales_copy['rank_in_region'] = sales.groupby('region')['sales'].transform('rank')
print(f"With rank:\n{sales_copy[['region', 'product', 'sales', 'rank_in_region']]}")
```

## Filter Groups

```python
print("\n=== FILTER GROUPS ===")

# filter() keeps groups meeting condition
# Keep only regions with total sales > 300
high_sales_regions = sales.groupby('region').filter(lambda x: x['sales'].sum() > 300)
print(f"Regions with total sales > 300:\n{high_sales_regions}\n")

# Keep groups with more than 1 product
multi_product = sales.groupby('region').filter(lambda x: len(x) > 1)
print(f"Regions with multiple products:\n{multi_product}")
```

## Iteration Over Groups

```python
print("\n=== ITERATING OVER GROUPS ===")

# Iterate through groups
for name, group in sales.groupby('region'):
    print(f"\nRegion: {name}")
    print(f"  Total sales: {group['sales'].sum()}")
    print(f"  Products: {group['product'].tolist()}")

# Get specific group
north_data = sales.groupby('region').get_group('North')
print(f"\nNorth region data:\n{north_data}")
```

## Time-Based GroupBy

```python
print("\n=== TIME-BASED GROUPBY ===")

# Create time series data
dates = pd.date_range('2024-01-01', periods=12, freq='M')
monthly_data = pd.DataFrame({
    'date': dates,
    'revenue': np.random.randint(1000, 5000, 12)
})
monthly_data['date'] = pd.to_datetime(monthly_data['date'])
print(f"Monthly data:\n{monthly_data}\n")

# Group by quarter
monthly_data['quarter'] = monthly_data['date'].dt.quarter
quarterly = monthly_data.groupby('quarter')['revenue'].sum()
print(f"Quarterly totals:\n{quarterly}")
```

## Key Points

- **groupby()**: Split data into groups based on column values
- **Aggregation**: `sum()`, `mean()`, `count()`, `min()`, `max()`, `std()`
- **agg()**: Apply multiple or custom aggregation functions
- **Named aggregations**: Use `agg(name=('col', 'func'))` for clear column names
- **transform()**: Apply function but return same-sized result
- **filter()**: Keep groups that meet a condition
- **Multiple groups**: `groupby(['col1', 'col2'])` for hierarchical grouping

## Reflection Questions

1. What's the difference between `agg()` and `transform()`?
2. When would you use `filter()` instead of boolean indexing?
3. How would you calculate the percentage of total for each group?
