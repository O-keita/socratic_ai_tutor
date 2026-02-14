# Functions and Control Flow

## Introduction

Functions allow you to organize code into reusable blocks, while control flow statements let you make decisions and handle different scenarios in your programs.

## Core Concepts

### Defining Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameter
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])
```

### Arguments and Parameters

```python
# Positional arguments
def add(a, b):
    return a + b

result = add(3, 5)

# Keyword arguments
result = add(a=3, b=5)

# *args for variable positional arguments
def sum_all(*args):
    return sum(args)

total = sum_all(1, 2, 3, 4, 5)

# **kwargs for variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25)
```

### Lambda Functions

Anonymous functions for simple operations:

```python
# Traditional function
def square(x):
    return x ** 2

# Lambda equivalent
square = lambda x: x ** 2

# Commonly used with map, filter, sorted
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

### Conditional Statements

```python
# if-elif-else
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary operator
status = "Pass" if score >= 60 else "Fail"
```

### Handling Exceptions

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always runs")

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

---

## Key Points

- Functions promote code reuse and organization
- Use default parameters for optional arguments
- `*args` and `**kwargs` handle variable arguments
- Lambda functions are concise for simple operations
- try-except blocks handle errors gracefully

---

## Reflection Questions

1. **Think**: How do functions help you write cleaner data analysis code? What makes a function well-designed?

2. **Consider**: When would you use a lambda function versus a regular function? What are the tradeoffs?

3. **Explore**: Why is exception handling important in data science? What errors might you encounter when loading or processing data?
