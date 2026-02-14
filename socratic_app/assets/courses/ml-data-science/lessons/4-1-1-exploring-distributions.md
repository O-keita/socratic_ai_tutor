# Exploring Distributions

## Introduction

Understanding data distributions is the foundation of exploratory data analysis. Before modeling, we must understand the shape, spread, and characteristics of our variables.

## Visualizing Single Variable Distributions

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

print("=== EXPLORING DISTRIBUTIONS ===")
print("""
Key Questions When Exploring Distributions:
  1. What is the center (typical value)?
  2. What is the spread (variability)?
  3. What is the shape (symmetric, skewed, multimodal)?
  4. Are there outliers?
  5. Are there any unusual patterns?

Common Distribution Shapes:
  - Symmetric/Normal: Bell-shaped, mean ≈ median
  - Right-skewed: Long tail to the right (income, prices)
  - Left-skewed: Long tail to the left (test scores at max)
  - Bimodal: Two peaks (mixed populations)
  - Uniform: Flat, equal probability
""")
```

## Summary Statistics for Distributions

```python
print("\n=== SUMMARY STATISTICS ===")

# Create sample distributions
normal_data = np.random.normal(100, 15, 1000)
skewed_data = np.random.exponential(50, 1000)
bimodal_data = np.concatenate([np.random.normal(30, 5, 500),
                               np.random.normal(70, 5, 500)])

def describe_distribution(data, name):
    """Compute comprehensive summary statistics."""
    print(f"\n{name}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}")
    print(f"  Range: {np.ptp(data):.2f}")  # peak-to-peak
    print(f"  IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
    print(f"  Skewness: {stats.skew(data):.2f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.2f}")
    
    # Interpret skewness
    skew = stats.skew(data)
    if abs(skew) < 0.5:
        shape = "approximately symmetric"
    elif skew > 0:
        shape = "right-skewed (positive skew)"
    else:
        shape = "left-skewed (negative skew)"
    print(f"  Shape: {shape}")

describe_distribution(normal_data, "Normal Distribution")
describe_distribution(skewed_data, "Right-Skewed Distribution")
describe_distribution(bimodal_data, "Bimodal Distribution")
```

## Percentiles and Quantiles

```python
print("\n=== PERCENTILES AND QUANTILES ===")

data = np.random.normal(100, 15, 1000)

print("Key Percentiles:")
percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    print(f"  {p}th percentile: {np.percentile(data, p):.2f}")

print("\nQuartiles (divide data into 4 parts):")
print(f"  Q1 (25th): {np.percentile(data, 25):.2f}")
print(f"  Q2 (50th, median): {np.percentile(data, 50):.2f}")
print(f"  Q3 (75th): {np.percentile(data, 75):.2f}")

print("\nDeciles (divide data into 10 parts):")
for i in range(10, 101, 10):
    print(f"  D{i//10}: {np.percentile(data, i):.2f}")
```

## Using Pandas for Distribution Analysis

```python
print("\n=== PANDAS DESCRIBE ===")

# Create sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.normal(35, 10, 500).clip(18, 70).astype(int),
    'income': np.random.exponential(50000, 500).clip(15000, 500000),
    'score': np.random.beta(8, 2, 500) * 100
})

print("DataFrame Description:")
print(df.describe())

print("\nAdditional Statistics:")
print(df.agg(['mean', 'median', 'std', 'skew']))

print("\nValue Counts for Age (binned):")
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100],
                         labels=['18-25', '26-35', '36-45', '46-55', '55+'])
print(df['age_group'].value_counts().sort_index())
```

## Detecting Distribution Shape

```python
print("\n=== DETECTING SHAPE ===")
print("""
Comparing Mean and Median:
  - Mean > Median: Right-skewed
  - Mean < Median: Left-skewed  
  - Mean ≈ Median: Symmetric

Skewness Coefficient:
  - |skew| < 0.5: Approximately symmetric
  - 0.5 < |skew| < 1: Moderately skewed
  - |skew| > 1: Highly skewed

Kurtosis (peakedness):
  - kurtosis = 0: Normal (mesokurtic)
  - kurtosis > 0: Heavy tails (leptokurtic)
  - kurtosis < 0: Light tails (platykurtic)
""")

def analyze_shape(data, name):
    """Analyze the shape of a distribution."""
    mean = np.mean(data)
    median = np.median(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    
    print(f"\n{name}:")
    print(f"  Mean: {mean:.2f}, Median: {median:.2f}")
    print(f"  Mean - Median: {mean - median:.2f}")
    print(f"  Skewness: {skew:.2f}")
    print(f"  Kurtosis: {kurt:.2f}")
    
    # Normality test
    stat, p_value = stats.normaltest(data)
    print(f"  Normality test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  → May be normally distributed")
    else:
        print("  → Likely NOT normally distributed")

# Test different distributions
analyze_shape(np.random.normal(0, 1, 1000), "Normal")
analyze_shape(np.random.exponential(1, 1000), "Exponential")
analyze_shape(np.random.uniform(0, 1, 1000), "Uniform")
```

## Binning and Histograms

```python
print("\n=== BINNING STRATEGIES ===")

data = np.random.normal(50, 15, 500)

print("Different Binning Approaches:")

# Fixed number of bins
n_bins = 20
bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
print(f"\n1. Fixed {n_bins} equal-width bins:")
print(f"   Bin width: {(bin_edges[1] - bin_edges[0]):.2f}")

# Sturges' rule
sturges = int(np.ceil(np.log2(len(data)) + 1))
print(f"\n2. Sturges' rule: {sturges} bins")

# Square root rule
sqrt_rule = int(np.ceil(np.sqrt(len(data))))
print(f"\n3. Square root rule: {sqrt_rule} bins")

# Freedman-Diaconis rule
iqr = np.percentile(data, 75) - np.percentile(data, 25)
fd_width = 2 * iqr / (len(data) ** (1/3))
fd_bins = int(np.ceil((data.max() - data.min()) / fd_width))
print(f"\n4. Freedman-Diaconis rule: {fd_bins} bins")

# Show histogram values
counts, edges = np.histogram(data, bins=10)
print(f"\nHistogram with 10 bins:")
print(f"  Bin edges: {np.round(edges, 1)}")
print(f"  Counts: {counts}")
```

## Empirical CDF

```python
print("\n=== EMPIRICAL CUMULATIVE DISTRIBUTION ===")

data = np.random.normal(100, 15, 200)
sorted_data = np.sort(data)

print("""
Empirical CDF:
  F(x) = proportion of data ≤ x
  
Interpretation:
  - F(90) = 0.25 means 25% of data is ≤ 90
  - Useful for comparing distributions
  - Steps at each data point
""")

# Calculate ECDF
ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

print(f"\nECDF values at key points:")
check_values = [70, 85, 100, 115, 130]
for val in check_values:
    ecdf_val = np.mean(sorted_data <= val)
    print(f"  F({val}) = {ecdf_val:.3f} ({ecdf_val*100:.1f}% of data ≤ {val})")

# Compare with theoretical normal
print(f"\nComparison with theoretical Normal(100, 15):")
for val in check_values:
    empirical = np.mean(sorted_data <= val)
    theoretical = stats.norm.cdf(val, 100, 15)
    print(f"  x={val}: Empirical={empirical:.3f}, Theoretical={theoretical:.3f}")
```

## Key Points

- **Center**: Mean for symmetric, median for skewed data
- **Spread**: Standard deviation, IQR, range
- **Shape**: Symmetric, skewed, or multimodal
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of tail heaviness
- **Percentiles**: Divide data into proportions
- **ECDF**: Cumulative proportion below each value

## Reflection Questions

1. Why might median be preferred over mean for skewed distributions?
2. How does sample size affect the reliability of distribution estimates?
3. What does a bimodal distribution suggest about the underlying data?
