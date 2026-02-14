# Introduction to NumPy Arrays

## Introduction

NumPy is the foundation of scientific computing in Python. Its array objects enable efficient numerical operations that are essential for data science and machine learning.

## Core Concepts

### What is NumPy?

NumPy (Numerical Python) provides:
- Efficient multi-dimensional arrays
- Fast mathematical operations
- Broadcasting capabilities
- Linear algebra functions

### Creating Arrays

```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Special arrays
zeros = np.zeros((3, 4))       # 3x4 array of zeros
ones = np.ones((2, 3))         # 2x3 array of ones
identity = np.eye(3)           # 3x3 identity matrix
empty = np.empty((2, 2))       # uninitialized array

# Sequences
range_arr = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]

# Random arrays
random_arr = np.random.rand(3, 3)  # uniform [0,1)
normal_arr = np.random.randn(3, 3) # standard normal
```

### Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)    # (2, 3) - dimensions
print(arr.ndim)     # 2 - number of dimensions
print(arr.size)     # 6 - total elements
print(arr.dtype)    # int64 - data type
```

### Why NumPy over Lists?

```python
# Speed comparison
import time

# Python list
py_list = list(range(1000000))
start = time.time()
result = [x * 2 for x in py_list]
print(f"List: {time.time() - start:.4f}s")

# NumPy array
np_arr = np.arange(1000000)
start = time.time()
result = np_arr * 2
print(f"NumPy: {time.time() - start:.4f}s")
# NumPy is typically 10-100x faster!
```

### Data Types

```python
# Specify dtype
float_arr = np.array([1, 2, 3], dtype=np.float64)
int_arr = np.array([1.5, 2.7, 3.9], dtype=np.int32)

# Convert dtype
converted = float_arr.astype(np.int32)
```

---

## Key Points

- NumPy arrays are homogeneous (single data type) and fixed-size
- Much faster than Python lists for numerical operations
- Foundation for pandas, scikit-learn, and TensorFlow
- Use `shape`, `dtype`, `ndim` to understand array structure
- Memory-efficient storage compared to Python lists

---

## Reflection Questions

1. **Think**: Why are NumPy arrays faster than Python lists? What tradeoffs come with this performance gain?

2. **Consider**: When would you use `np.zeros()` vs `np.empty()`? What's the practical difference?

3. **Explore**: How does specifying the correct dtype affect memory usage and computation? When might you use float32 vs float64?
