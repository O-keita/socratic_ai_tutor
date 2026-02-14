# Central Tendency and Spread

## Introduction

Central tendency and spread are the two fundamental ways to summarize a distribution. Central tendency tells us where the "center" of the data lies, while spread (dispersion) tells us how much the data varies around that center.

## Measures of Central Tendency

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

print("=== MEASURES OF CENTRAL TENDENCY ===")

# Sample data with different characteristics
symmetric_data = np.array([45, 47, 48, 50, 50, 50, 52, 53, 55])
skewed_data = np.array([20, 25, 30, 35, 40, 45, 50, 100, 200])
bimodal_data = np.array([10, 12, 14, 15, 50, 52, 54, 55, 56])

print("Three datasets:")
print(f"  Symmetric: {symmetric_data}")
print(f"  Right-skewed: {skewed_data}")
print(f"  Bimodal: {bimodal_data}")

def central_tendency(data, name):
    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data, keepdims=True)
    mode = mode_result.mode[0]
    
    print(f"\n{name}:")
    print(f"  Mean:   {mean:.1f}")
    print(f"  Median: {median:.1f}")
    print(f"  Mode:   {mode}")

central_tendency(symmetric_data, "Symmetric")
central_tendency(skewed_data, "Right-skewed")
central_tendency(bimodal_data, "Bimodal")
```

## When to Use Each Measure

```python
print("\n=== CHOOSING THE RIGHT MEASURE ===")
print("""
USE MEAN WHEN:
  ✓ Data is approximately symmetric
  ✓ No significant outliers
  ✓ You need to use it in further calculations
  ✓ Interval or ratio data
  
USE MEDIAN WHEN:
  ✓ Data is skewed
  ✓ Outliers are present
  ✓ You want the "typical" value
  ✓ Ordinal data or better
  
USE MODE WHEN:
  ✓ Data is categorical (nominal)
  ✓ You want the most common value
  ✓ Data has multiple peaks (report all modes)
  ✓ Any type of data

EXAMPLE: Income Data
  Mean income: $85,000 (pulled up by millionaires)
  Median income: $52,000 (more representative)
  → Median is better for typical household
""")

# Demonstrate with income-like data
incomes = np.array([30000, 40000, 45000, 50000, 55000, 60000, 75000, 250000, 500000])
print(f"\nIncome example: {incomes}")
print(f"  Mean:   ${np.mean(incomes):,.0f}")
print(f"  Median: ${np.median(incomes):,.0f}")
print(f"  The median better represents typical income!")
```

## Weighted Mean

```python
print("\n=== WEIGHTED MEAN ===")

# Grades with different weights
grades = np.array([85, 90, 78, 92])
weights = np.array([0.2, 0.3, 0.1, 0.4])  # Sum to 1

print("Course grades with weights:")
print(f"  Grades: {grades}")
print(f"  Weights: {weights}")

# Regular mean
regular_mean = np.mean(grades)

# Weighted mean
weighted_mean = np.average(grades, weights=weights)

print(f"\nRegular mean: {regular_mean:.1f}")
print(f"Weighted mean: {weighted_mean:.1f}")
print(f"  Formula: Σ(weight × value) / Σ(weights)")
print(f"  = ({weights[0]}×{grades[0]} + {weights[1]}×{grades[1]} + {weights[2]}×{grades[2]} + {weights[3]}×{grades[3]})")
print(f"  = {weighted_mean:.1f}")
```

## Measures of Spread

```python
print("\n=== MEASURES OF SPREAD ===")

data = np.array([10, 20, 25, 30, 35, 40, 50])
print(f"Data: {data}")
print(f"Mean: {np.mean(data):.1f}")

# Range
data_range = np.max(data) - np.min(data)
print(f"\n1. RANGE: {data_range}")
print(f"   Max - Min = {np.max(data)} - {np.min(data)} = {data_range}")
print(f"   Pros: Simple to calculate")
print(f"   Cons: Very sensitive to outliers")

# Interquartile Range (IQR)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
print(f"\n2. IQR: {iqr}")
print(f"   Q3 - Q1 = {q3} - {q1} = {iqr}")
print(f"   Middle 50% of data")
print(f"   Pros: Robust to outliers")

# Variance
variance = np.var(data, ddof=1)  # Sample variance
print(f"\n3. VARIANCE: {variance:.2f}")
print(f"   Average squared deviation from mean")
print(f"   Units are squared (hard to interpret)")

# Standard Deviation
std_dev = np.std(data, ddof=1)  # Sample std dev
print(f"\n4. STANDARD DEVIATION: {std_dev:.2f}")
print(f"   Square root of variance")
print(f"   Same units as data (easy to interpret)")
```

## Calculating Variance and Standard Deviation

```python
print("\n=== VARIANCE CALCULATION STEP-BY-STEP ===")

data = np.array([4, 8, 6, 5, 3, 2, 8, 9, 2, 5])
print(f"Data: {data}")

# Step 1: Calculate mean
mean = np.mean(data)
print(f"\n1. Mean = {mean}")

# Step 2: Calculate deviations
deviations = data - mean
print(f"2. Deviations from mean: {deviations}")

# Step 3: Square the deviations
squared_devs = deviations ** 2
print(f"3. Squared deviations: {squared_devs}")

# Step 4: Calculate variance (sample variance uses n-1)
n = len(data)
sample_variance = np.sum(squared_devs) / (n - 1)
print(f"4. Sample Variance = Σ(x-μ)² / (n-1) = {np.sum(squared_devs):.2f} / {n-1} = {sample_variance:.2f}")

# Step 5: Standard deviation
std_dev = np.sqrt(sample_variance)
print(f"5. Standard Deviation = √Variance = {std_dev:.2f}")

# Verify with numpy
print(f"\nVerification:")
print(f"  np.var(data, ddof=1) = {np.var(data, ddof=1):.2f}")
print(f"  np.std(data, ddof=1) = {np.std(data, ddof=1):.2f}")
```

## Population vs Sample Statistics

```python
print("\n=== POPULATION VS SAMPLE ===")
print("""
POPULATION (N):
  - All members of a group
  - Parameters: μ (mean), σ² (variance), σ (std dev)
  - Variance formula: σ² = Σ(x-μ)² / N
  
SAMPLE (n):
  - Subset of population
  - Statistics: x̄ (mean), s² (variance), s (std dev)
  - Variance formula: s² = Σ(x-x̄)² / (n-1)
  
Why n-1 (Bessel's correction)?
  - Sample variance tends to underestimate population variance
  - Dividing by n-1 gives unbiased estimate
  - Called "degrees of freedom" correction
""")

data = np.array([5, 7, 3, 8, 6])
print(f"Data: {data}")
print(f"Population variance (ddof=0): {np.var(data, ddof=0):.4f}")
print(f"Sample variance (ddof=1):     {np.var(data, ddof=1):.4f}")
```

## Coefficient of Variation

```python
print("\n=== COEFFICIENT OF VARIATION ===")

# Compare variability across different scales
heights = np.array([165, 170, 175, 168, 172])  # cm
weights = np.array([60, 75, 80, 65, 70])  # kg

print("Comparing variability:")
print(f"Heights: mean={np.mean(heights):.1f}cm, std={np.std(heights, ddof=1):.1f}cm")
print(f"Weights: mean={np.mean(weights):.1f}kg, std={np.std(weights, ddof=1):.1f}kg")

# Coefficient of Variation = (std / mean) × 100%
cv_heights = (np.std(heights, ddof=1) / np.mean(heights)) * 100
cv_weights = (np.std(weights, ddof=1) / np.mean(weights)) * 100

print(f"\nCoefficient of Variation (CV):")
print(f"  Heights: {cv_heights:.1f}%")
print(f"  Weights: {cv_weights:.1f}%")
print(f"\nWeights have higher relative variability!")
print(f"CV allows comparison across different units/scales")
```

## The Empirical Rule (68-95-99.7)

```python
print("\n=== EMPIRICAL RULE (Normal Distribution) ===")
print("""
For approximately normal distributions:

  ±1σ from mean: ~68% of data
  ±2σ from mean: ~95% of data
  ±3σ from mean: ~99.7% of data

       68%
    ◄──────►
    ┌──────┐
  ▄▄█      █▄▄
 ▄██        ██▄
▄███        ███▄
████        ████
───┼────┼────┼───
  μ-σ   μ   μ+σ
  
     ◄─────95%─────►
""")

# Demonstrate with normal data
normal_data = np.random.normal(100, 15, 10000)
mean = np.mean(normal_data)
std = np.std(normal_data)

within_1std = np.sum((normal_data >= mean - std) & (normal_data <= mean + std)) / len(normal_data) * 100
within_2std = np.sum((normal_data >= mean - 2*std) & (normal_data <= mean + 2*std)) / len(normal_data) * 100
within_3std = np.sum((normal_data >= mean - 3*std) & (normal_data <= mean + 3*std)) / len(normal_data) * 100

print(f"Normal data (n=10000, μ={mean:.1f}, σ={std:.1f}):")
print(f"  Within ±1σ: {within_1std:.1f}% (expected ~68%)")
print(f"  Within ±2σ: {within_2std:.1f}% (expected ~95%)")
print(f"  Within ±3σ: {within_3std:.1f}% (expected ~99.7%)")
```

## Key Points

- **Mean**: Arithmetic average, sensitive to outliers
- **Median**: Middle value, robust to outliers
- **Mode**: Most frequent value, works for any data type
- **Range**: Max - Min, very sensitive to outliers
- **IQR**: Q3 - Q1, robust measure of spread
- **Variance**: Average squared deviation, hard to interpret
- **Std Dev**: Square root of variance, same units as data
- **CV**: Relative measure, allows comparison across scales
- **Use sample formulas (n-1) for samples from populations**

## Reflection Questions

1. Why does the sample variance formula use n-1 instead of n?
2. When would you report both mean and median together?
3. How does the empirical rule help identify outliers?
