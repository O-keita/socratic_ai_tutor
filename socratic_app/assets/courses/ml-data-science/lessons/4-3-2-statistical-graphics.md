# Statistical Graphics and Chart Types

## Introduction

This lesson covers the implementation and proper use of common statistical graphics, with practical Python examples using matplotlib and seaborn.

## Distribution Plots

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=== HISTOGRAMS ===")
print("""
Purpose: Show distribution of a single variable
When to use:
  - Understanding data spread
  - Identifying skewness, modality
  - Detecting outliers
  
Key parameters:
  - bins: Number of bars (affects interpretation)
  - density: Normalize to show proportions
""")

# Create sample data
data = np.concatenate([
    np.random.normal(30, 5, 500),
    np.random.normal(50, 8, 300)
])

# Histogram code example
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].hist(data, bins=10, edgecolor='black')
axes[0].set_title('10 bins')

axes[1].hist(data, bins=30, edgecolor='black')
axes[1].set_title('30 bins')

axes[2].hist(data, bins=50, edgecolor='black')
axes[2].set_title('50 bins')

plt.suptitle('Effect of Bin Count on Histogram')
plt.tight_layout()
plt.show()
```

## Box Plots and Violin Plots

```python
print("\n=== BOX PLOTS ===")
print("""
Components:
  - Box: IQR (25th to 75th percentile)
  - Line in box: Median
  - Whiskers: 1.5 * IQR from box
  - Points beyond whiskers: Outliers

Best for:
  - Comparing distributions across groups
  - Identifying outliers
  - Showing spread and center
""")

# Create grouped data
groups = ['A', 'B', 'C', 'D']
group_data = {
    'A': np.random.normal(50, 10, 100),
    'B': np.random.normal(60, 15, 100),
    'C': np.random.normal(45, 8, 100),
    'D': np.random.exponential(20, 100) + 30
}

df = pd.DataFrame(group_data).melt(var_name='Group', value_name='Value')

# Box plot code
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Basic box plot
sns.boxplot(data=df, x='Group', y='Value', ax=axes[0])
axes[0].set_title('Box Plot')

# Violin plot (shows distribution shape)
sns.violinplot(data=df, x='Group', y='Value', ax=axes[1])
axes[1].set_title('Violin Plot')

plt.tight_layout()
plt.show()

print("""
Violin plots show:
  - Full distribution shape
  - Multimodality
  - Better for larger samples
""")
```

## Scatter Plots

```python
print("\n=== SCATTER PLOTS ===")
print("""
Purpose: Show relationship between two continuous variables
Enhancements:
  - Color: Third variable (categorical or continuous)
  - Size: Fourth variable
  - Transparency (alpha): Handle overplotting
  - Regression line: Show trend
""")

# Generate correlated data
n = 200
x = np.random.uniform(0, 100, n)
y = 2*x + np.random.normal(0, 30, n)
category = np.random.choice(['Type A', 'Type B'], n)
size_var = np.random.uniform(10, 100, n)

df_scatter = pd.DataFrame({
    'x': x, 'y': y, 'category': category, 'size': size_var
})

# Scatter plot variations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic scatter
axes[0, 0].scatter(df_scatter['x'], df_scatter['y'], alpha=0.5)
axes[0, 0].set_title('Basic Scatter')

# With color by category
for cat in ['Type A', 'Type B']:
    mask = df_scatter['category'] == cat
    axes[0, 1].scatter(df_scatter.loc[mask, 'x'], 
                       df_scatter.loc[mask, 'y'], 
                       label=cat, alpha=0.5)
axes[0, 1].legend()
axes[0, 1].set_title('Colored by Category')

# With regression line
axes[1, 0].scatter(df_scatter['x'], df_scatter['y'], alpha=0.5)
z = np.polyfit(df_scatter['x'], df_scatter['y'], 1)
p = np.poly1d(z)
axes[1, 0].plot(sorted(df_scatter['x']), p(sorted(df_scatter['x'])), 
                'r-', linewidth=2)
axes[1, 0].set_title('With Regression Line')

# Bubble chart (size varies)
axes[1, 1].scatter(df_scatter['x'], df_scatter['y'], 
                   s=df_scatter['size'], alpha=0.5)
axes[1, 1].set_title('Bubble Chart (size = third variable)')

plt.tight_layout()
plt.show()
```

## Bar Charts

```python
print("\n=== BAR CHARTS ===")
print("""
Purpose: Compare values across categories
Types:
  - Vertical bars: Standard comparison
  - Horizontal bars: Long category names
  - Grouped bars: Compare subcategories
  - Stacked bars: Show composition
  
Rules:
  - Always start y-axis at zero!
  - Order bars meaningfully (by value or logically)
""")

# Sample data
categories = ['Product A', 'Product B', 'Product C', 'Product D']
values = [45, 72, 38, 55]
subcategories = {
    'Q1': [20, 35, 15, 25],
    'Q2': [25, 37, 23, 30]
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic bar
axes[0, 0].bar(categories, values)
axes[0, 0].set_title('Basic Bar Chart')
axes[0, 0].set_ylabel('Sales')

# Horizontal bar (sorted)
sorted_idx = np.argsort(values)
axes[0, 1].barh([categories[i] for i in sorted_idx], 
                [values[i] for i in sorted_idx])
axes[0, 1].set_title('Horizontal (Sorted)')

# Grouped bar
x = np.arange(len(categories))
width = 0.35
axes[1, 0].bar(x - width/2, subcategories['Q1'], width, label='Q1')
axes[1, 0].bar(x + width/2, subcategories['Q2'], width, label='Q2')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].legend()
axes[1, 0].set_title('Grouped Bar Chart')

# Stacked bar
axes[1, 1].bar(categories, subcategories['Q1'], label='Q1')
axes[1, 1].bar(categories, subcategories['Q2'], 
               bottom=subcategories['Q1'], label='Q2')
axes[1, 1].legend()
axes[1, 1].set_title('Stacked Bar Chart')

plt.tight_layout()
plt.show()
```

## Line Charts

```python
print("\n=== LINE CHARTS ===")
print("""
Purpose: Show trends over ordered dimension (usually time)
Best practices:
  - Use for continuous/connected data
  - Consider multiple lines for comparison
  - Add markers for discrete data points
  - Use area fill to show magnitude
""")

# Time series data
dates = pd.date_range('2023-01-01', periods=12, freq='M')
series1 = np.cumsum(np.random.randn(12)) + 50
series2 = np.cumsum(np.random.randn(12)) + 45

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic line
axes[0, 0].plot(dates, series1)
axes[0, 0].set_title('Basic Line Chart')
axes[0, 0].tick_params(axis='x', rotation=45)

# Multiple lines
axes[0, 1].plot(dates, series1, label='Product A', marker='o')
axes[0, 1].plot(dates, series2, label='Product B', marker='s')
axes[0, 1].legend()
axes[0, 1].set_title('Multiple Lines')
axes[0, 1].tick_params(axis='x', rotation=45)

# Area chart
axes[1, 0].fill_between(dates, series1, alpha=0.5)
axes[1, 0].plot(dates, series1)
axes[1, 0].set_title('Area Chart')
axes[1, 0].tick_params(axis='x', rotation=45)

# Stacked area
axes[1, 1].stackplot(dates, series1, series2, labels=['A', 'B'], alpha=0.7)
axes[1, 1].legend(loc='upper left')
axes[1, 1].set_title('Stacked Area Chart')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Heatmaps

```python
print("\n=== HEATMAPS ===")
print("""
Purpose: Show values in a matrix format
Use cases:
  - Correlation matrices
  - Time × Category patterns
  - Confusion matrices
  - Geographic data (with proper projection)
""")

# Correlation matrix
np.random.seed(42)
df_corr = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'D': np.random.randn(100),
    'E': np.random.randn(100)
})
df_corr['B'] = df_corr['A'] * 0.7 + np.random.randn(100) * 0.3
df_corr['E'] = -df_corr['C'] * 0.5 + np.random.randn(100) * 0.5

corr_matrix = df_corr.corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Basic heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=axes[0])
axes[0].set_title('Correlation Heatmap')

# Custom color map (diverging for correlation)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('With Diverging Color Scale')

plt.tight_layout()
plt.show()
```

## Subplots and Faceting

```python
print("\n=== FACETING / SMALL MULTIPLES ===")
print("""
Purpose: Show same visualization across subgroups
Advantages:
  - Easy comparison across groups
  - Avoids overcrowded single plot
  - Shows patterns within and across groups
""")

# Create grouped data
np.random.seed(42)
df_facet = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C', 'D'], 100),
    'x': np.random.uniform(0, 100, 400),
    'y': np.tile([np.random.normal(50, 10, 100),
                  np.random.normal(60, 15, 100),
                  np.random.normal(45, 8, 100),
                  np.random.normal(55, 12, 100)], 1).flatten()
})

# FacetGrid example
g = sns.FacetGrid(df_facet, col='category', col_wrap=2, height=4)
g.map(plt.hist, 'y', bins=20, edgecolor='black')
g.set_titles('{col_name}')
plt.suptitle('Distribution by Category', y=1.02)
plt.show()

# Pair plot for multiple variables
df_multi = pd.DataFrame({
    'Var1': np.random.normal(50, 10, 100),
    'Var2': np.random.normal(100, 20, 100),
    'Var3': np.random.normal(25, 5, 100),
    'Group': np.random.choice(['A', 'B'], 100)
})
df_multi['Var2'] = df_multi['Var1'] * 1.5 + np.random.normal(0, 10, 100)

sns.pairplot(df_multi, hue='Group')
plt.suptitle('Pair Plot', y=1.02)
plt.show()
```

## Key Points

- **Histograms**: Bin choice affects interpretation
- **Box plots**: Show median, IQR, outliers
- **Scatter plots**: Encode multiple variables with position, color, size
- **Bar charts**: Always start at zero, order meaningfully
- **Line charts**: Best for time series and trends
- **Heatmaps**: Use appropriate color scales
- **Faceting**: Compare patterns across groups

## Reflection Questions

1. When would you choose a violin plot over a box plot?
2. How does the choice of bin width affect histogram interpretation?
3. What are the advantages of small multiples over a single complex chart?
