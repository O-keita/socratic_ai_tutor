# Dictionaries and Sets

## Introduction

Dictionaries and sets are essential Python data structures. Dictionaries provide key-value mappings perfect for structured data, while sets handle unique collections efficiently.

## Core Concepts

### Dictionaries

Key-value pairs for storing related data:

```python
# Creating dictionaries
student = {
    "name": "Alice",
    "age": 22,
    "major": "Data Science",
    "gpa": 3.8
}

# Empty dictionary
empty_dict = {}
# or
empty_dict = dict()
```

### Accessing Dictionary Values

```python
# Access by key
name = student["name"]  # "Alice"

# Safe access with get()
major = student.get("major")  # "Data Science"
phone = student.get("phone", "Not provided")  # default value

# Check if key exists
if "age" in student:
    print(student["age"])
```

### Modifying Dictionaries

```python
# Add or update
student["email"] = "alice@example.com"
student["age"] = 23

# Remove items
del student["gpa"]
removed = student.pop("email")

# Get all keys, values, items
keys = student.keys()
values = student.values()
items = student.items()
```

### Dictionary Comprehensions

```python
# Create dictionary from lists
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
grade_book = {name: score for name, score in zip(names, scores)}

# With condition
passing = {k: v for k, v in grade_book.items() if v >= 80}
```

### Sets

Collections of unique elements:

```python
# Creating sets
fruits = {"apple", "banana", "cherry"}
numbers = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# Sets automatically remove duplicates
unique_values = set([1, 1, 2, 2, 3])  # {1, 2, 3}
```

### Set Operations

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# Union (all elements)
union = a | b  # {1, 2, 3, 4, 5, 6}

# Intersection (common elements)
intersection = a & b  # {3, 4}

# Difference (in a but not b)
difference = a - b  # {1, 2}

# Symmetric difference (in one but not both)
sym_diff = a ^ b  # {1, 2, 5, 6}
```

---

## Key Points

- Dictionaries store key-value pairs with O(1) lookup time
- Keys must be immutable (strings, numbers, tuples)
- Use `.get()` for safe access with default values
- Sets contain only unique elements
- Set operations (union, intersection, difference) are powerful for data analysis

---

## Reflection Questions

1. **Think**: When would you choose a dictionary over a list? What advantages does the key-value structure provide for organizing data?

2. **Consider**: How can sets help you find unique values in a dataset? What operations would you use to find common customers between two lists?

3. **Explore**: Why must dictionary keys be immutable? What would happen if you could use a list as a key?
