# Data Inspection with Pandas

## Introduction

Data inspection is the first step in any data analysis project. Before building models or drawing conclusions, you must understand your data's structure, types, distributions, and potential issues. Pandas provides powerful tools for exploring and understanding datasets.

## Loading and First Look

```python
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', None, 'Grace'],
    'age': [25, 30, None, 28, 35, 42, 29, 31],
    'salary': [50000, 60000, 55000, None, 75000, 80000, 52000, 58000],
    'department': ['Sales', 'Engineering', 'Sales', 'Marketing', 'Engineering', 'Sales', 'Marketing', 'Engineering'],
    'hire_date': pd.to_datetime(['2020-01-15', '2019-06-20', '2021-03-10', '2020-08-05', '2018-11-30', '2017-04-22', '2022-01-05', '2021-07-18']),
    'is_manager': [False, True, False, False, True, True, False, False]
}
df = pd.DataFrame(data)

print("=== FIRST LOOK AT DATA ===")
print("\n1. View first few rows:")
print(df.head())

print("\n2. View last few rows:")
print(df.tail(3))

print("\n3. Random sample:")
print(df.sample(3, random_state=42))
```

## Understanding Data Structure

```python
print("\n=== DATA STRUCTURE ===")

print("\n1. Shape (rows, columns):")
print(f"   {df.shape}")
print(f"   {df.shape[0]} rows, {df.shape[1]} columns")

print("\n2. Column names:")
print(f"   {df.columns.tolist()}")

print("\n3. Data types:")
print(df.dtypes)

print("\n4. Detailed info:")
print(df.info())
```

## The info() Method Deep Dive

```python
print("\n=== INFO() EXPLAINED ===")
print("""
df.info() shows:
  - Total entries (rows)
  - Column names
  - Non-null count per column
  - Data type per column
  - Memory usage

Key insights:
  - Non-null count < total rows → missing values
  - Object type often means strings
  - Memory usage helps with large datasets
""")

# Detect columns with missing values
print("\nColumns with missing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nMissing value percentage:")
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_pct[missing_pct > 0])
```

## Statistical Summary with describe()

```python
print("\n=== STATISTICAL SUMMARY ===")

print("\n1. Numeric columns (default):")
print(df.describe())

print("\n2. All columns (including categorical):")
print(df.describe(include='all'))

print("\n3. Only object/categorical columns:")
print(df.describe(include=['object']))

print("""
describe() statistics explained:
  count  - Non-null values
  mean   - Average value
  std    - Standard deviation (spread)
  min    - Minimum value
  25%    - First quartile (Q1)
  50%    - Median (Q2)
  75%    - Third quartile (Q3)
  max    - Maximum value
  
For categorical:
  unique - Number of distinct values
  top    - Most frequent value
  freq   - Frequency of most common value
""")
```

## Inspecting Individual Columns

```python
print("\n=== COLUMN INSPECTION ===")

print("\n1. Unique values:")
print(f"   Departments: {df['department'].unique()}")
print(f"   Number of unique: {df['department'].nunique()}")

print("\n2. Value counts (frequency):")
print(df['department'].value_counts())

print("\n3. Value counts with percentages:")
print(df['department'].value_counts(normalize=True).round(3) * 100)

print("\n4. Including NaN in counts:")
print(df['age'].value_counts(dropna=False))
```

## Data Type Inspection

```python
print("\n=== DATA TYPE INSPECTION ===")

print("\n1. Check specific types:")
print(f"   Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"   Object columns: {df.select_dtypes(include=['object']).columns.tolist()}")
print(f"   Datetime columns: {df.select_dtypes(include=['datetime64']).columns.tolist()}")
print(f"   Boolean columns: {df.select_dtypes(include=['bool']).columns.tolist()}")

print("\n2. Memory usage by column:")
print(df.memory_usage(deep=True))

print("\n3. Total memory:")
print(f"   {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
```

## Detecting Data Quality Issues

```python
print("\n=== DATA QUALITY CHECKS ===")

print("\n1. Missing values summary:")
missing_summary = pd.DataFrame({
    'missing_count': df.isnull().sum(),
    'missing_percent': (df.isnull().sum() / len(df) * 100).round(2),
    'dtype': df.dtypes
})
print(missing_summary[missing_summary['missing_count'] > 0])

print("\n2. Duplicate rows:")
print(f"   Total duplicates: {df.duplicated().sum()}")
print(f"   Duplicate rows:\n{df[df.duplicated(keep=False)]}")

print("\n3. Check for potential outliers (numeric):")
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
    if len(outliers) > 0:
        print(f"   {col}: {len(outliers)} potential outliers")
```

## Correlation Analysis

```python
print("\n=== CORRELATION ANALYSIS ===")

print("\n1. Correlation matrix:")
print(df.select_dtypes(include=[np.number]).corr().round(3))

print("\n2. Correlation with specific column:")
numeric_df = df.select_dtypes(include=[np.number])
if 'salary' in numeric_df.columns:
    print(numeric_df.corr()['salary'].sort_values(ascending=False))
```

## Practical Inspection Workflow

```python
print("\n=== RECOMMENDED INSPECTION WORKFLOW ===")
print("""
1. LOAD & PREVIEW
   df = pd.read_csv('data.csv')
   df.head()
   df.tail()
   df.sample(5)

2. STRUCTURE
   df.shape
   df.columns
   df.dtypes
   df.info()

3. STATISTICS
   df.describe()
   df.describe(include='all')

4. MISSING VALUES
   df.isnull().sum()
   df.isnull().sum() / len(df) * 100

5. UNIQUE VALUES
   df['column'].unique()
   df['column'].nunique()
   df['column'].value_counts()

6. DUPLICATES
   df.duplicated().sum()
   df[df.duplicated()]

7. CORRELATIONS
   df.corr()

8. DATA TYPES
   df.select_dtypes(include=['number'])
   df.select_dtypes(include=['object'])
""")
```

## Quick Data Profiling Function

```python
def quick_profile(df):
    """Generate a quick profile of a DataFrame."""
    print("=" * 50)
    print("QUICK DATA PROFILE")
    print("=" * 50)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print(f"\nDuplicates: {df.duplicated().sum()} rows")
    
    print("\nColumn Summary:")
    print("-" * 50)
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"  {col}:")
        print(f"    Type: {dtype}, Missing: {missing}, Unique: {unique}")
    
    print("\nNumeric Columns Stats:")
    print(df.describe().round(2))

# Run the profile
quick_profile(df)
```

## Key Points

- **head()/tail()**: Preview first/last rows
- **shape**: Get dimensions (rows, columns)
- **info()**: Data types, non-null counts, memory
- **describe()**: Statistical summary of columns
- **value_counts()**: Frequency of unique values
- **isnull().sum()**: Count missing values
- **duplicated()**: Find duplicate rows
- **dtypes**: Check column data types

## Reflection Questions

1. Why should you inspect data before any analysis or modeling?
2. What does a large difference between mean and median suggest?
3. How would you decide which columns need cleaning based on info() output?
