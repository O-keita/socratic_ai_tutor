# Data Summaries and Aggregation

## Introduction

Effective data summaries condense large datasets into meaningful statistics. This lesson covers techniques for summarizing data at various levels of granularity.

## Basic Summary Statistics

```python
import numpy as np
import pandas as pd

np.random.seed(42)

print("=== DATA SUMMARIES ===")

# Create sample sales data
n = 1000
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n, freq='D'),
    'product': np.random.choice(['A', 'B', 'C', 'D'], n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'sales': np.random.exponential(500, n),
    'units': np.random.poisson(10, n),
    'discount': np.random.uniform(0, 0.3, n)
})
df['revenue'] = df['sales'] * df['units'] * (1 - df['discount'])

print("Sample Data Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\n=== DESCRIBE ===")
print(df.describe())
```

## Aggregating Numeric Data

```python
print("\n=== AGGREGATION FUNCTIONS ===")

print("Single Aggregations:")
print(f"  Total revenue: ${df['revenue'].sum():,.2f}")
print(f"  Average sale: ${df['sales'].mean():.2f}")
print(f"  Median units: {df['units'].median():.0f}")
print(f"  Max discount: {df['discount'].max():.2%}")

print("\nMultiple Aggregations:")
agg_result = df['revenue'].agg(['sum', 'mean', 'median', 'std', 'min', 'max'])
print(agg_result)

print("\nCustom Aggregations:")
def coefficient_of_variation(x):
    return x.std() / x.mean()

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

print(f"  Revenue CV: {coefficient_of_variation(df['revenue']):.3f}")
print(f"  Revenue IQR: ${iqr(df['revenue']):,.2f}")
```

## GroupBy Operations

```python
print("\n=== GROUPBY SUMMARIES ===")

# Group by single column
print("Revenue by Product:")
print(df.groupby('product')['revenue'].agg(['sum', 'mean', 'count']))

print("\nRevenue by Region:")
print(df.groupby('region')['revenue'].agg(['sum', 'mean', 'count']))

# Group by multiple columns
print("\nRevenue by Product and Region:")
print(df.groupby(['product', 'region'])['revenue'].mean().unstack())

# Multiple aggregations on multiple columns
print("\nComprehensive Summary:")
summary = df.groupby('product').agg({
    'revenue': ['sum', 'mean', 'std'],
    'units': ['sum', 'mean'],
    'discount': 'mean'
})
print(summary)
```

## Cross-Tabulation

```python
print("\n=== CROSS-TABULATION ===")

# Add a categorical column for analysis
df['sales_tier'] = pd.cut(df['sales'], bins=[0, 200, 500, 1000, float('inf')],
                          labels=['Low', 'Medium', 'High', 'Premium'])

# Simple crosstab
print("Product vs Sales Tier (counts):")
print(pd.crosstab(df['product'], df['sales_tier']))

print("\nProduct vs Region (counts):")
print(pd.crosstab(df['product'], df['region']))

# With margins (totals)
print("\nWith Margins:")
print(pd.crosstab(df['product'], df['region'], margins=True))

# Proportions
print("\nProportions (row-wise):")
print(pd.crosstab(df['product'], df['region'], normalize='index').round(3))
```

## Pivot Tables

```python
print("\n=== PIVOT TABLES ===")

# Basic pivot table
print("Average Revenue by Product and Region:")
pivot1 = df.pivot_table(values='revenue', 
                        index='product', 
                        columns='region', 
                        aggfunc='mean')
print(pivot1.round(2))

# Multiple aggregations
print("\nRevenue: Mean and Sum:")
pivot2 = df.pivot_table(values='revenue',
                        index='product',
                        aggfunc=['mean', 'sum'])
print(pivot2.round(2))

# Multiple values and aggregations
print("\nComprehensive Pivot Table:")
pivot3 = df.pivot_table(values=['revenue', 'units'],
                        index='product',
                        columns='region',
                        aggfunc='sum',
                        fill_value=0)
print(pivot3.round(0))
```

## Time-Based Summaries

```python
print("\n=== TIME-BASED AGGREGATION ===")

# Set date as index
df_ts = df.set_index('date')

# Monthly summary
print("Monthly Revenue Summary:")
monthly = df_ts.resample('M')['revenue'].agg(['sum', 'mean', 'count'])
print(monthly.head())

# Weekly average
print("\nWeekly Average Sales:")
weekly = df_ts.resample('W')['sales'].mean()
print(weekly.head())

# Quarterly summary
print("\nQuarterly Summary:")
quarterly = df_ts.resample('Q').agg({
    'revenue': 'sum',
    'units': 'sum',
    'sales': 'mean'
})
print(quarterly)
```

## Rolling and Expanding Statistics

```python
print("\n=== ROLLING STATISTICS ===")

# Daily revenue
daily_revenue = df_ts['revenue'].resample('D').sum()

# 7-day rolling average
print("7-Day Rolling Average (first 10 days):")
rolling_7 = daily_revenue.rolling(window=7).mean()
print(rolling_7.head(10))

# 30-day rolling statistics
print("\n30-Day Rolling Statistics:")
rolling_30 = daily_revenue.rolling(window=30).agg(['mean', 'std', 'min', 'max'])
print(rolling_30.head(5))

print("\n=== EXPANDING STATISTICS ===")
# Cumulative statistics
print("Cumulative Sum (Running Total):")
cumsum = daily_revenue.expanding().sum()
print(cumsum.head(10))

# Cumulative mean
print("\nCumulative Mean:")
cummean = daily_revenue.expanding().mean()
print(cummean.head(10))
```

## Ranking and Percentiles

```python
print("\n=== RANKING ===")

# Rank within groups
df['revenue_rank'] = df.groupby('product')['revenue'].rank(ascending=False)
print("Top 3 sales per product:")
print(df[df['revenue_rank'] <= 3][['product', 'revenue', 'revenue_rank']].sort_values(['product', 'revenue_rank']))

# Percentile rank
df['revenue_pctile'] = df['revenue'].rank(pct=True)
print(f"\nRevenue Percentile Distribution:")
print(df['revenue_pctile'].describe())

# Binned percentiles
df['revenue_quartile'] = pd.qcut(df['revenue'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print("\nQuartile Distribution:")
print(df['revenue_quartile'].value_counts().sort_index())
```

## Summary Tables

```python
print("\n=== CREATING SUMMARY TABLES ===")

def create_summary_table(df, group_col, value_col):
    """Create a comprehensive summary table."""
    summary = df.groupby(group_col)[value_col].agg([
        ('Count', 'count'),
        ('Sum', 'sum'),
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('Median', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        ('Max', 'max')
    ]).round(2)
    return summary

print("Revenue Summary by Product:")
print(create_summary_table(df, 'product', 'revenue'))

print("\nRevenue Summary by Region:")
print(create_summary_table(df, 'region', 'revenue'))
```

## Key Points

- **describe()**: Quick overview of numeric columns
- **agg()**: Apply multiple functions to columns
- **groupby()**: Split-apply-combine for group summaries
- **crosstab()**: Frequency tables for categorical data
- **pivot_table()**: Flexible multi-dimensional aggregation
- **resample()**: Time-based aggregation
- **rolling()**: Moving window statistics
- **expanding()**: Cumulative statistics

## Reflection Questions

1. When would you use a pivot table versus groupby operations?
2. How do rolling averages help identify trends in time series data?
3. What summary statistics are most appropriate for skewed distributions?
