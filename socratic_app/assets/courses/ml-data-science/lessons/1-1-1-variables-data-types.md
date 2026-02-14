# Variables and Data Types in Python

## Introduction

Variables are the fundamental building blocks of any programming language. In Python, variables act as containers that store data values, and understanding data types is essential for effective data science work.

## Core Concepts

### What is a Variable?

A variable is a named location in memory that holds a value. Unlike some languages, Python doesn't require you to declare the type of a variable—it's dynamically typed.

```python
# Creating variables
name = "Alice"
age = 25
height = 5.6
is_student = True
```

### Python's Core Data Types

1. **Numeric Types**
   - `int`: Whole numbers (42, -7, 0)
   - `float`: Decimal numbers (3.14, -0.001)
   - `complex`: Complex numbers (3+4j)

2. **Text Type**
   - `str`: Strings ("Hello", 'World')

3. **Boolean Type**
   - `bool`: True or False

4. **None Type**
   - `None`: Represents absence of value

### Type Conversion

Python allows converting between types:

```python
# Convert string to integer
num_str = "42"
num_int = int(num_str)

# Convert integer to float
x = float(10)  # 10.0

# Convert to string
text = str(3.14)  # "3.14"
```

---

## Key Points

- Variables store values and can be reassigned
- Python is dynamically typed—types are inferred
- Use `type()` to check a variable's type
- Type conversion functions: `int()`, `float()`, `str()`, `bool()`
- Variable names should be descriptive and follow naming conventions

---

## Reflection Questions

1. **Think**: If Python is dynamically typed, what happens when you assign a string to a variable that previously held an integer? How does this flexibility affect your coding?

2. **Consider**: When would you need to explicitly convert between data types? What errors might occur if types don't match?

3. **Explore**: What's the difference between `0`, `0.0`, `"0"`, and `False` in Python? How might confusing these affect your data analysis?
