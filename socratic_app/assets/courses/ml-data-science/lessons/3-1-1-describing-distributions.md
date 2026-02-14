# Describing Distributions

## Introduction

Understanding data distributions is fundamental to statistics and data science. A distribution describes how values in a dataset are spread across different possible values, revealing patterns, central tendencies, and variability.

## Visualizing Distributions

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate sample data
normal_data = np.random.normal(loc=50, scale=10, size=1000)
skewed_data = np.random.exponential(scale=10, size=1000)
bimodal_data = np.concatenate([
    np.random.normal(30, 5, 500),
    np.random.normal(70, 5, 500)
])

print("=== DISTRIBUTION SHAPES ===")
print("""
Common Distribution Shapes:

1. SYMMETRIC (Normal/Bell-Shaped)
   - Mean ≈ Median ≈ Mode
   - Equal tails on both sides
   - Example: Heights, test scores
   
        ▄█▄
       ▄███▄
      ▄█████▄
     ▄███████▄
   ▄▄▄▄▄▄▄▄▄▄▄▄▄

2. RIGHT-SKEWED (Positive Skew)
   - Mean > Median > Mode
   - Long tail to the right
   - Example: Income, home prices
   
   █▄
   ██▄▄
   █████▄▄▄▄▄▄▄

3. LEFT-SKEWED (Negative Skew)
   - Mean < Median < Mode
   - Long tail to the left
   - Example: Age at retirement
   
             ▄█
          ▄▄██
   ▄▄▄▄▄█████

4. BIMODAL
   - Two distinct peaks
   - Two different groups in data
   - Example: Heights (men+women combined)
   
     ▄█▄     ▄█▄
    ▄███▄   ▄███▄
   ▄█████▄▄▄█████▄
""")
```

## Describing Center: Mean, Median, Mode

```python
print("\n=== MEASURES OF CENTER ===")

data = [12, 15, 18, 18, 20, 22, 25, 28, 30, 100]
print(f"Data: {data}\n")

# Mean (average)
mean = np.mean(data)
print(f"Mean: {mean}")
print(f"  Formula: Sum / Count = {sum(data)} / {len(data)} = {mean}")

# Median (middle value)
median = np.median(data)
print(f"\nMedian: {median}")
print(f"  Middle value when sorted")
print(f"  Robust to outliers (100 doesn't affect it much)")

# Mode (most frequent)
from scipy import stats
mode_result = stats.mode(data, keepdims=True)
print(f"\nMode: {mode_result.mode[0]}")
print(f"  Most frequent value")

print(f"""
Comparison with outlier (100):
  Mean ({mean}) is pulled toward the outlier
  Median ({median}) is resistant to the outlier
  
When to use which:
  - Mean: Symmetric distributions, no outliers
  - Median: Skewed data, outliers present
  - Mode: Categorical data, finding most common
""")
```

## Describing Spread: Variance and Standard Deviation

```python
print("\n=== MEASURES OF SPREAD ===")

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(f"Data: {data}")
print(f"Mean: {np.mean(data)}")

# Variance: Average squared deviation from mean
mean = np.mean(data)
deviations = data - mean
squared_devs = deviations ** 2
variance = np.mean(squared_devs)  # Population variance

print(f"\nVariance calculation:")
print(f"  Deviations from mean: {deviations}")
print(f"  Squared deviations: {squared_devs}")
print(f"  Variance (σ²): {variance}")

# Standard deviation: Square root of variance
std_dev = np.sqrt(variance)
print(f"  Standard Deviation (σ): {std_dev:.2f}")

# NumPy functions
print(f"\nUsing NumPy:")
print(f"  np.var(data): {np.var(data):.2f}")
print(f"  np.std(data): {np.std(data):.2f}")
print(f"  np.var(data, ddof=1): {np.var(data, ddof=1):.2f}  # Sample variance")
print(f"  np.std(data, ddof=1): {np.std(data, ddof=1):.2f}  # Sample std dev")

print(f"""
Interpretation:
  - Low std dev = data clustered near mean
  - High std dev = data spread out from mean
  - Same units as original data (unlike variance)
""")
```

## Range and Interquartile Range (IQR)

```python
print("\n=== RANGE AND IQR ===")

data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
print(f"Data: {data}")

# Range
data_range = np.max(data) - np.min(data)
print(f"\nRange: {data_range}")
print(f"  Max - Min = {np.max(data)} - {np.min(data)} = {data_range}")
print(f"  Sensitive to outliers")

# Quartiles
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # Median
q3 = np.percentile(data, 75)
iqr = q3 - q1

print(f"\nQuartiles:")
print(f"  Q1 (25th percentile): {q1}")
print(f"  Q2 (50th percentile): {q2} (median)")
print(f"  Q3 (75th percentile): {q3}")
print(f"  IQR (Q3 - Q1): {iqr}")

# Five-number summary
print(f"\nFive-Number Summary:")
print(f"  Min: {np.min(data)}")
print(f"  Q1:  {q1}")
print(f"  Median: {q2}")
print(f"  Q3:  {q3}")
print(f"  Max: {np.max(data)}")

print(f"""
IQR is robust to outliers:
  - Contains middle 50% of data
  - Used for outlier detection: value < Q1-1.5*IQR or > Q3+1.5*IQR
  - Lower fence: {q1 - 1.5*iqr}
  - Upper fence: {q3 + 1.5*iqr}
  - 100 is above upper fence → outlier!
""")
```

## Percentiles and Quantiles

```python
print("\n=== PERCENTILES ===")

scores = np.array([65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98])
print(f"Test scores: {scores}")

# Specific percentiles
p10 = np.percentile(scores, 10)
p50 = np.percentile(scores, 50)
p90 = np.percentile(scores, 90)

print(f"\nPercentiles:")
print(f"  10th percentile: {p10}")
print(f"  50th percentile: {p50} (median)")
print(f"  90th percentile: {p90}")

# What percentile is a specific score?
score = 85
percentile_rank = (scores < score).sum() / len(scores) * 100
print(f"\nScore of {score} is at the {percentile_rank:.0f}th percentile")
print(f"  (Better than {percentile_rank:.0f}% of scores)")

# Deciles (10 equal parts)
deciles = np.percentile(scores, [10, 20, 30, 40, 50, 60, 70, 80, 90])
print(f"\nDeciles: {deciles}")
```

## Using Pandas describe()

```python
print("\n=== PANDAS DESCRIBE() ===")

df = pd.DataFrame({
    'height': np.random.normal(170, 10, 100),
    'weight': np.random.normal(70, 15, 100),
    'age': np.random.randint(20, 60, 100)
})

print("Summary statistics:")
print(df.describe())

print(f"""
describe() provides:
  count - number of non-null values
  mean  - average
  std   - standard deviation
  min   - minimum value
  25%   - first quartile (Q1)
  50%   - median (Q2)
  75%   - third quartile (Q3)
  max   - maximum value
""")
```

## Skewness and Kurtosis

```python
print("\n=== SKEWNESS AND KURTOSIS ===")

from scipy.stats import skew, kurtosis

# Different distributions
symmetric = np.random.normal(0, 1, 1000)
right_skewed = np.random.exponential(1, 1000)
left_skewed = -np.random.exponential(1, 1000)

print("Skewness (asymmetry):")
print(f"  Symmetric data: {skew(symmetric):.3f} (≈ 0)")
print(f"  Right-skewed: {skew(right_skewed):.3f} (> 0)")
print(f"  Left-skewed: {skew(left_skewed):.3f} (< 0)")

print(f"""
Skewness interpretation:
  = 0: Symmetric
  > 0: Right-skewed (tail extends right)
  < 0: Left-skewed (tail extends left)

Kurtosis (tailedness):
  = 0: Normal distribution (mesokurtic)
  > 0: Heavy tails (leptokurtic)
  < 0: Light tails (platykurtic)
""")

print(f"\nKurtosis (excess, relative to normal):")
print(f"  Normal: {kurtosis(symmetric):.3f}")
print(f"  Exponential: {kurtosis(right_skewed):.3f}")
```

## Key Points

- **Mean, Median, Mode**: Measures of central tendency
- **Variance and Std Dev**: Measures of spread around the mean
- **Range and IQR**: Simple spread measures (IQR is robust)
- **Percentiles**: Divide data into 100 equal parts
- **Skewness**: Measures asymmetry of distribution
- **Kurtosis**: Measures tail heaviness
- **Use median and IQR for skewed data with outliers**

## Reflection Questions

1. Why is the median preferred over the mean for income data?
2. How does adding an outlier affect mean vs median?
3. What does a high standard deviation tell you about data variability?
