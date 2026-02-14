# Lists and Collections in Python

## Introduction

Lists are one of Python's most versatile data structures. They allow you to store collections of items, which is essential when working with datasets in data science.

## Core Concepts

### Creating Lists

```python
# Empty list
empty_list = []

# List with values
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List from range
sequence = list(range(10))  # [0, 1, 2, ..., 9]
```

### Accessing Elements

Lists use zero-based indexing:

```python
fruits = ["apple", "banana", "cherry", "date"]

# Positive indexing
first = fruits[0]   # "apple"
third = fruits[2]   # "cherry"

# Negative indexing (from end)
last = fruits[-1]   # "date"
second_last = fruits[-2]  # "cherry"
```

### Slicing

Extract portions of a list:

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slicing [start:end]
first_three = numbers[0:3]  # [0, 1, 2]
middle = numbers[3:7]       # [3, 4, 5, 6]

# With step [start:end:step]
evens = numbers[0:10:2]     # [0, 2, 4, 6, 8]
reversed_list = numbers[::-1]  # [9, 8, 7, ..., 0]
```

### List Operations

```python
# Append (add to end)
fruits.append("elderberry")

# Insert at position
fruits.insert(1, "blueberry")

# Remove by value
fruits.remove("banana")

# Remove by index
del fruits[0]
popped = fruits.pop()  # removes and returns last item

# Concatenation
combined = [1, 2] + [3, 4]  # [1, 2, 3, 4]

# Length
length = len(fruits)
```

### List Comprehensions

Powerful way to create lists:

```python
# Traditional approach
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension
squares = [x**2 for x in range(10)]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

---

## Key Points

- Lists are ordered, mutable collections
- Zero-based indexing; negative indices count from end
- Slicing creates new lists without modifying original
- List comprehensions provide concise syntax
- Lists can contain mixed types (though not recommended for data analysis)

---

## Reflection Questions

1. **Think**: Why is understanding indexing crucial when working with data? What happens if you try to access an index that doesn't exist?

2. **Consider**: How do list comprehensions compare to traditional loops in terms of readability and performance? When might each approach be preferred?

3. **Explore**: If lists are mutable, what implications does this have when you assign one list to another variable? What's the difference between copying and referencing?
