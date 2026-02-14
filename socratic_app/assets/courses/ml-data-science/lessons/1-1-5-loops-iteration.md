# Loops and Iteration

## Introduction

Loops are essential for processing data, applying operations to collections, and automating repetitive tasks—core activities in data science workflows.

## Core Concepts

### For Loops

Iterate over sequences:

```python
# Loop through a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Loop with index using enumerate
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Loop through dictionary
student = {"name": "Alice", "age": 22}
for key, value in student.items():
    print(f"{key}: {value}")

# Loop through range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4
```

### While Loops

Loop while condition is true:

```python
count = 0
while count < 5:
    print(count)
    count += 1

# With break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == "quit":
        break
```

### Loop Control

```python
# continue - skip current iteration
for i in range(10):
    if i % 2 == 0:
        continue  # skip even numbers
    print(i)

# break - exit loop
for i in range(10):
    if i == 5:
        break  # stop at 5
    print(i)

# else clause (runs if no break)
for i in range(5):
    print(i)
else:
    print("Loop completed without break")
```

### Iterating Multiple Sequences

```python
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]

# zip combines sequences
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Creating pairs
pairs = list(zip(names, scores))
# [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
```

### Nested Loops

```python
# Matrix iteration
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # new line after each row
```

### Efficient Iteration Patterns

```python
# Using itertools for advanced iteration
from itertools import combinations, permutations

# All pairs
items = [1, 2, 3]
for combo in combinations(items, 2):
    print(combo)  # (1,2), (1,3), (2,3)

# Generator expressions (memory efficient)
sum_of_squares = sum(x**2 for x in range(1000000))
```

---

## Key Points

- `for` loops iterate over sequences; `while` loops run until condition is false
- `enumerate()` provides index alongside values
- `zip()` combines multiple sequences element-wise
- `break` exits loop; `continue` skips to next iteration
- Generator expressions are memory-efficient for large datasets

---

## Reflection Questions

1. **Think**: When processing large datasets, why might a generator expression be preferable to a list comprehension?

2. **Consider**: How would you iterate through a dataset to find all records meeting certain criteria? What's the most readable approach?

3. **Explore**: What happens if you modify a list while iterating over it? How can you safely remove items from a collection during iteration?
