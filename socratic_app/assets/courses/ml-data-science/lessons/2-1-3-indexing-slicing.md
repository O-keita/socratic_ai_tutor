# NumPy Indexing and Slicing

## Introduction

Indexing and slicing are fundamental operations for accessing and manipulating data in NumPy arrays. Mastering these techniques is essential for efficient data manipulation in data science workflows.

## Basic Indexing

```python
import numpy as np

np.random.seed(42)

# 1D Array Indexing
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

print("=== 1D ARRAY INDEXING ===")
print(f"Array: {arr}")
print(f"First element (index 0): {arr[0]}")
print(f"Last element (index -1): {arr[-1]}")
print(f"Third element (index 2): {arr[2]}")
print(f"Second to last (index -2): {arr[-2]}")
```

## Slicing 1D Arrays

```python
print("\n=== 1D SLICING ===")
print(f"Array: {arr}")

# Basic slicing: arr[start:stop:step]
print(f"\nFirst 3 elements [0:3]: {arr[0:3]}")
print(f"Elements 2-5 [2:6]: {arr[2:6]}")
print(f"Last 3 elements [-3:]: {arr[-3:]}")
print(f"All except last 2 [:-2]: {arr[:-2]}")
print(f"Every other element [::2]: {arr[::2]}")
print(f"Reverse array [::-1]: {arr[::-1]}")
print(f"Every 3rd from index 1 [1::3]: {arr[1::3]}")

# Omitting indices
print(f"\nFrom start to 5 [:5]: {arr[:5]}")
print(f"From 3 to end [3:]: {arr[3:]}")
print(f"All elements [:]: {arr[:]}")
```

## 2D Array Indexing

```python
print("\n=== 2D ARRAY INDEXING ===")

# Create 2D array
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix:\n{matrix}")

# Access elements: matrix[row, column]
print(f"\nElement at row 0, col 0: {matrix[0, 0]}")
print(f"Element at row 1, col 2: {matrix[1, 2]}")
print(f"Element at row 2, col 3: {matrix[2, 3]}")
print(f"Last row, last col: {matrix[-1, -1]}")

# Access entire rows/columns
print(f"\nFirst row [0, :]: {matrix[0, :]}")
print(f"Second column [:, 1]: {matrix[:, 1]}")
print(f"Last row [-1, :]: {matrix[-1, :]}")
```

## 2D Array Slicing

```python
print("\n=== 2D SLICING ===")
print(f"Matrix:\n{matrix}")

# Slice rows and columns
print(f"\nFirst 2 rows [:2, :]:\n{matrix[:2, :]}")
print(f"\nLast 2 columns [:, -2:]:\n{matrix[:, -2:]}")
print(f"\nSubmatrix [0:2, 1:3]:\n{matrix[0:2, 1:3]}")
print(f"\nEvery other row [::2, :]:\n{matrix[::2, :]}")
print(f"\nEvery other column [:, ::2]:\n{matrix[:, ::2]}")
```

## Boolean Indexing

```python
print("\n=== BOOLEAN INDEXING ===")

data = np.array([15, 25, 35, 45, 55, 65, 75])
print(f"Data: {data}")

# Create boolean mask
mask = data > 40
print(f"\nMask (data > 40): {mask}")
print(f"Elements > 40: {data[mask]}")

# Direct boolean indexing
print(f"\nElements > 30: {data[data > 30]}")
print(f"Elements <= 50: {data[data <= 50]}")
print(f"Elements between 20 and 60: {data[(data > 20) & (data < 60)]}")
print(f"Elements < 20 or > 60: {data[(data < 20) | (data > 60)]}")

# Boolean indexing with 2D arrays
print(f"\nMatrix:\n{matrix}")
print(f"Elements > 6: {matrix[matrix > 6]}")
```

## Fancy Indexing

```python
print("\n=== FANCY INDEXING ===")

arr = np.array([10, 20, 30, 40, 50, 60, 70])
print(f"Array: {arr}")

# Index with array of indices
indices = [0, 2, 4, 6]
print(f"\nIndices: {indices}")
print(f"Elements at indices: {arr[indices]}")

# Can have duplicates
print(f"Indices [1, 1, 3, 3]: {arr[[1, 1, 3, 3]]}")

# Fancy indexing with 2D arrays
print(f"\nMatrix:\n{matrix}")
rows = [0, 2]
cols = [1, 3]
print(f"Elements at (rows={rows}, cols={cols}): {matrix[rows, cols]}")

# Select specific rows
print(f"Rows 0 and 2:\n{matrix[[0, 2], :]}")
```

## np.where() for Conditional Indexing

```python
print("\n=== np.where() ===")

scores = np.array([75, 82, 90, 65, 88, 72, 95])
print(f"Scores: {scores}")

# Find indices where condition is true
passing_indices = np.where(scores >= 80)
print(f"\nIndices where score >= 80: {passing_indices[0]}")
print(f"Passing scores: {scores[passing_indices]}")

# Conditional assignment
grades = np.where(scores >= 80, 'Pass', 'Fail')
print(f"Grades: {grades}")

# Multiple conditions
letter_grades = np.where(scores >= 90, 'A',
                np.where(scores >= 80, 'B',
                np.where(scores >= 70, 'C', 'F')))
print(f"Letter grades: {letter_grades}")
```

## Modifying Arrays with Indexing

```python
print("\n=== MODIFYING ARRAYS ===")

arr = np.array([1, 2, 3, 4, 5])
print(f"Original: {arr}")

# Modify single element
arr[0] = 100
print(f"After arr[0] = 100: {arr}")

# Modify slice
arr[1:4] = [200, 300, 400]
print(f"After slice modification: {arr}")

# Modify with boolean indexing
arr[arr > 200] = 999
print(f"After boolean modification: {arr}")

# Modify 2D array
matrix_copy = matrix.copy()
matrix_copy[0, :] = 0  # Zero out first row
print(f"\nMatrix after zeroing first row:\n{matrix_copy}")
```

## Views vs Copies

```python
print("\n=== VIEWS VS COPIES ===")

original = np.array([1, 2, 3, 4, 5])
print(f"Original: {original}")

# Slicing creates a VIEW (shares memory)
view = original[1:4]
view[0] = 999
print(f"\nAfter modifying view:")
print(f"  View: {view}")
print(f"  Original: {original}")  # Original changed too!

# Use .copy() to create independent copy
original = np.array([1, 2, 3, 4, 5])
copy = original[1:4].copy()
copy[0] = 999
print(f"\nAfter modifying copy:")
print(f"  Copy: {copy}")
print(f"  Original: {original}")  # Original unchanged
```

## Key Points

- **Basic indexing**: `arr[index]` for single elements, negative indices count from end
- **Slicing**: `arr[start:stop:step]`, stop is exclusive
- **2D indexing**: `matrix[row, column]` or `matrix[row_slice, col_slice]`
- **Boolean indexing**: `arr[arr > value]` selects elements matching condition
- **Fancy indexing**: `arr[[0, 2, 4]]` selects elements at specific indices
- **np.where()**: Find indices or perform conditional assignment
- **Views vs copies**: Slices are views; use `.copy()` for independent copies

## Reflection Questions

1. Why does NumPy use exclusive stop indices in slicing?
2. When would you use boolean indexing vs fancy indexing?
3. Why is understanding views vs copies important for memory management?
