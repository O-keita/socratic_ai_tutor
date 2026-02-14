# Array Operations and Broadcasting

## Introduction

NumPy's true power lies in vectorized operations and broadcasting—techniques that enable efficient computation without explicit loops.

## Core Concepts

### Element-wise Operations

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Arithmetic operations
print(a + b)    # [6, 8, 10, 12]
print(a - b)    # [-4, -4, -4, -4]
print(a * b)    # [5, 12, 21, 32]
print(a / b)    # [0.2, 0.33, 0.43, 0.5]
print(a ** 2)   # [1, 4, 9, 16]

# Comparison operations
print(a > 2)    # [False, False, True, True]
print(a == b)   # [False, False, False, False]
```

### Broadcasting

NumPy can operate on arrays of different shapes:

```python
# Scalar broadcasting
arr = np.array([1, 2, 3, 4])
result = arr * 10  # [10, 20, 30, 40]

# Array broadcasting
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row = np.array([10, 20, 30])

result = matrix + row
# [[11, 22, 33],
#  [14, 25, 36]]

# Column broadcasting
column = np.array([[100], [200]])
result = matrix + column
# [[101, 102, 103],
#  [204, 205, 206]]
```

### Broadcasting Rules

1. Arrays are compared from right to left
2. Dimensions are compatible if equal or one is 1
3. Arrays with fewer dimensions are padded with 1s on the left

```python
# Shape (3,4) and (4,) -> Compatible
# Shape (3,4) and (3,) -> NOT compatible
# Shape (3,4) and (1,4) -> Compatible
# Shape (3,4) and (3,1) -> Compatible
```

### Aggregation Functions

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Global aggregations
print(np.sum(arr))      # 21
print(np.mean(arr))     # 3.5
print(np.std(arr))      # standard deviation
print(np.min(arr))      # 1
print(np.max(arr))      # 6

# Along axis
print(np.sum(arr, axis=0))  # [5, 7, 9] - column sums
print(np.sum(arr, axis=1))  # [6, 15] - row sums
print(np.mean(arr, axis=0)) # column means
```

### Universal Functions (ufuncs)

```python
arr = np.array([1, 4, 9, 16])

# Mathematical functions
print(np.sqrt(arr))     # [1, 2, 3, 4]
print(np.exp(arr))      # exponential
print(np.log(arr))      # natural log
print(np.sin(arr))      # sine

# Useful ufuncs
print(np.abs(np.array([-1, -2, 3])))  # [1, 2, 3]
print(np.round(np.array([1.4, 1.6]))) # [1, 2]
```

---

## Key Points

- Vectorized operations eliminate need for explicit loops
- Broadcasting allows operations between different-shaped arrays
- Axis parameter controls direction of aggregation (0=columns, 1=rows)
- Universal functions apply operations element-wise
- Broadcasting follows specific rules about shape compatibility

---

## Reflection Questions

1. **Think**: How does broadcasting help normalize data? For example, how would you subtract the mean of each column from a matrix?

2. **Consider**: What happens if you try to add arrays with shapes (3,4) and (2,4)? Why does broadcasting fail here?

3. **Explore**: When computing the mean along axis=0, why does the result have fewer dimensions than the input? How does `keepdims=True` change this?
