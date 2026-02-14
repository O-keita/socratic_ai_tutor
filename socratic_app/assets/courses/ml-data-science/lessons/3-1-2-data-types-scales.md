# Data Types and Measurement Scales

## Introduction

Understanding data types and measurement scales is crucial for choosing appropriate statistical methods and visualizations. Different types of data require different analytical approaches.

## The Four Measurement Scales

```python
import pandas as pd
import numpy as np

np.random.seed(42)

print("=== FOUR MEASUREMENT SCALES ===")
print("""
NIVEAU OF MEASUREMENT (Stevens, 1946):

1. NOMINAL (Categorical/Names)
   - Categories with no order
   - Examples: Gender, Color, Country
   - Operations: = ≠ (equality only)
   - Stats: Mode, frequency counts

2. ORDINAL (Ranked/Ordered)
   - Categories WITH meaningful order
   - But intervals between categories not equal
   - Examples: Education level, Satisfaction rating
   - Operations: = ≠ < > (order)
   - Stats: Mode, median, percentiles

3. INTERVAL (Equal Intervals)
   - Numeric with equal intervals
   - No true zero point
   - Examples: Temperature (°C, °F), Year
   - Operations: + - (addition, subtraction)
   - Stats: Mean, std dev, correlation

4. RATIO (True Zero)
   - Numeric with equal intervals AND true zero
   - Examples: Height, Weight, Income, Age
   - Operations: × ÷ (multiplication, ratios)
   - Stats: All statistics, geometric mean

Memory aid: NOIR (French for "black")
  N - Nominal
  O - Ordinal
  I - Interval
  R - Ratio
""")
```

## Nominal Data

```python
print("\n=== NOMINAL DATA ===")

# Example nominal data
customers = pd.DataFrame({
    'customer_id': range(1, 11),
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'M'],
    'country': ['USA', 'UK', 'USA', 'Canada', 'UK', 'USA', 'Canada', 'USA', 'UK', 'USA'],
    'blood_type': ['A', 'B', 'O', 'AB', 'A', 'O', 'B', 'A', 'O', 'A']
})

print(f"Nominal data example:\n{customers}\n")

# Appropriate analysis for nominal data
print("Appropriate analyses:")
print(f"  Mode (most common country): {customers['country'].mode()[0]}")
print(f"\n  Frequency counts:\n{customers['country'].value_counts()}")
print(f"\n  Proportions:\n{customers['country'].value_counts(normalize=True)}")

print("""
What you CANNOT do with nominal data:
  ✗ Calculate mean (average gender makes no sense)
  ✗ Sort in meaningful order
  ✗ Perform arithmetic operations
""")
```

## Ordinal Data

```python
print("\n=== ORDINAL DATA ===")

# Example ordinal data
survey = pd.DataFrame({
    'respondent': range(1, 11),
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 
                  'High School', 'Master', 'Bachelor', 'PhD', 'Bachelor'],
    'satisfaction': ['Poor', 'Good', 'Excellent', 'Good', 'Fair',
                     'Poor', 'Good', 'Excellent', 'Good', 'Fair']
})

print(f"Ordinal data example:\n{survey}\n")

# Define order for ordinal categories
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
satisfaction_order = ['Poor', 'Fair', 'Good', 'Excellent']

# Convert to ordered categorical
survey['education'] = pd.Categorical(survey['education'], 
                                      categories=education_order, 
                                      ordered=True)
survey['satisfaction'] = pd.Categorical(survey['satisfaction'], 
                                         categories=satisfaction_order, 
                                         ordered=True)

print("After ordering:")
print(f"  Education levels: {survey['education'].cat.categories.tolist()}")
print(f"  Satisfaction levels: {survey['satisfaction'].cat.categories.tolist()}")

# Appropriate analysis
print(f"\nMedian education: {survey['education'].median()}")  # Requires numeric codes
print(f"\nSatisfaction distribution:\n{survey['satisfaction'].value_counts().sort_index()}")

print("""
What you CAN do with ordinal data:
  ✓ Identify mode and median
  ✓ Calculate percentiles
  ✓ Compare (greater than, less than)
  
What you CANNOT reliably do:
  ✗ Calculate mean (intervals not equal)
  ✗ Add/subtract values
""")
```

## Interval Data

```python
print("\n=== INTERVAL DATA ===")

# Example interval data
weather = pd.DataFrame({
    'day': range(1, 11),
    'temp_celsius': [20, 22, 25, 23, 21, 19, 18, 20, 24, 26],
    'year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022, 2020]
})

print(f"Interval data example:\n{weather}\n")

# Appropriate analysis
print("Temperature analysis:")
print(f"  Mean: {weather['temp_celsius'].mean():.1f}°C")
print(f"  Std Dev: {weather['temp_celsius'].std():.1f}°C")
print(f"  Range: {weather['temp_celsius'].min()} to {weather['temp_celsius'].max()}°C")

# Can calculate differences
diff = weather['temp_celsius'].max() - weather['temp_celsius'].min()
print(f"  Difference: {diff}°C")

print("""
Key characteristic: No true zero!
  - 0°C doesn't mean "no temperature"
  - 20°C is NOT "twice as hot" as 10°C
  
What you CAN do:
  ✓ Calculate mean, std dev
  ✓ Add and subtract values
  ✓ Calculate correlation
  
What you CANNOT do:
  ✗ Form meaningful ratios (20°C ÷ 10°C ≠ "twice as warm")
""")
```

## Ratio Data

```python
print("\n=== RATIO DATA ===")

# Example ratio data
measurements = pd.DataFrame({
    'person': range(1, 11),
    'height_cm': [165, 180, 172, 158, 190, 175, 168, 182, 170, 177],
    'weight_kg': [65, 80, 72, 55, 95, 78, 62, 85, 68, 75],
    'income': [45000, 75000, 52000, 38000, 120000, 65000, 48000, 85000, 55000, 72000]
})

print(f"Ratio data example:\n{measurements}\n")

# All statistical operations valid
print("All statistics valid:")
print(f"  Mean height: {measurements['height_cm'].mean():.1f} cm")
print(f"  Median income: ${measurements['income'].median():,.0f}")
print(f"  Std dev weight: {measurements['weight_kg'].std():.1f} kg")

# Ratios are meaningful
ratio = measurements['height_cm'].max() / measurements['height_cm'].min()
print(f"\nMeaningful ratios:")
print(f"  Tallest/Shortest: {ratio:.2f}x taller")
print(f"  Person with 0 kg would have no weight (true zero)")

# Coefficient of variation
cv = measurements['income'].std() / measurements['income'].mean() * 100
print(f"  Coefficient of Variation (income): {cv:.1f}%")
```

## Discrete vs Continuous

```python
print("\n=== DISCRETE VS CONTINUOUS ===")

print("""
DISCRETE DATA:
  - Countable, finite values
  - Often integers
  - Examples:
    • Number of children (0, 1, 2, 3...)
    • Dice roll (1, 2, 3, 4, 5, 6)
    • Number of defects
    • Count of purchases
    
CONTINUOUS DATA:
  - Infinite possible values in a range
  - Measurable to any precision
  - Examples:
    • Height (165.0, 165.1, 165.11...)
    • Time (can be infinitely precise)
    • Temperature
    • Distance

This is DIFFERENT from measurement scales!
  - Discrete can be nominal, ordinal, or ratio
  - Continuous is usually interval or ratio
""")

# Examples
data = pd.DataFrame({
    'num_children': [2, 0, 3, 1, 2],  # Discrete ratio
    'height': [165.5, 180.2, 172.8, 158.1, 190.0],  # Continuous ratio
    'satisfaction_1_5': [4, 3, 5, 2, 4],  # Discrete ordinal
    'temperature': [20.5, 22.3, 19.8, 21.1, 23.7]  # Continuous interval
})
print(f"Mixed data types:\n{data}")
```

## Identifying Data Types in Practice

```python
print("\n=== IDENTIFYING DATA TYPES ===")

# Sample dataset
df = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'age': [25, 34, 45, 28, 52],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'education': ['Bachelor', 'Master', 'PhD', 'High School', 'Master'],
    'satisfaction': [4, 5, 3, 4, 2],  # 1-5 scale
    'purchase_amount': [150.50, 230.00, 89.99, 320.00, 175.50],
    'num_purchases': [3, 7, 2, 5, 4]
})

print(f"Dataset:\n{df}\n")

print("Data Type Analysis:")
print("""
  customer_id:     Nominal (identifier, no order)
  age:             Ratio (true zero, ratios meaningful)
  gender:          Nominal (categories, no order)
  education:       Ordinal (ordered categories)
  satisfaction:    Ordinal (1-5 scale, ordered but intervals may not be equal)
  purchase_amount: Ratio (true zero, continuous)
  num_purchases:   Ratio (true zero, discrete count)
""")

# Pandas data types
print("Pandas inferred dtypes:")
print(df.dtypes)
```

## Key Points

- **Nominal**: Categories without order (mode, frequency)
- **Ordinal**: Ordered categories (median, percentiles)
- **Interval**: Equal intervals, no true zero (mean, std dev)
- **Ratio**: True zero, all operations valid (ratios, geometric mean)
- **Discrete vs Continuous**: Countable vs measurable
- **Choose statistics based on data type!**

## Reflection Questions

1. Why can't you calculate a meaningful average of ZIP codes?
2. Is a 1-5 satisfaction scale ordinal or interval? Why does it matter?
3. What measurement scale is temperature in Kelvin vs Celsius?
