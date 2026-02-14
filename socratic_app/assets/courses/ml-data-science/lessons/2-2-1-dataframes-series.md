# DataFrames and Series

## Introduction

Pandas is the cornerstone of data manipulation in Python. Its DataFrame and Series objects provide intuitive, powerful tools for handling structured data.

## Core Concepts

### Series: One-Dimensional Data

```python
import pandas as pd
import numpy as np

# Create a Series
s = pd.Series([10, 20, 30, 40])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40

# With custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s['b'])  # 20

# From dictionary
data = {'apple': 3, 'banana': 5, 'cherry': 2}
fruits = pd.Series(data)
```

### DataFrame: Two-Dimensional Data

```python
# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
}
df = pd.DataFrame(data)

# From list of dictionaries
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
]
df = pd.DataFrame(data)

# From NumPy array
arr = np.random.randn(3, 4)
df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])
```

### Reading Data

```python
# CSV files
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv', 
                 sep=',',
                 header=0,
                 index_col='id',
                 parse_dates=['date_column'])

# Excel files
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json')
```

### Saving Data

```python
# To CSV
df.to_csv('output.csv', index=False)

# To Excel
df.to_excel('output.xlsx', sheet_name='Results')

# To JSON
df.to_json('output.json', orient='records')
```

### Basic Attributes

```python
# Shape and size
print(df.shape)      # (rows, columns)
print(len(df))       # number of rows
print(df.columns)    # column names
print(df.index)      # row index
print(df.dtypes)     # data types of each column

# Quick info
df.info()            # summary of DataFrame
df.describe()        # statistical summary
```

---

## Key Points

- Series = 1D labeled array; DataFrame = 2D labeled table
- DataFrames have both row index and column names
- Read data from CSV, Excel, JSON, SQL, and more
- Use `df.info()` and `df.describe()` for quick exploration
- DataFrames are built on NumPy arrays for efficiency

---

## Reflection Questions

1. **Think**: Why might you choose to set a specific column as the index when reading a CSV file? What advantages does this provide?

2. **Consider**: How does pandas handle missing values when reading data? What options do you have for dealing with them?

3. **Explore**: What's the difference between `df.shape[0]` and `len(df)`? When might they give different results?
