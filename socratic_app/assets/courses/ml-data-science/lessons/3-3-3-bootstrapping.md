# Bootstrapping and Confidence Intervals

## Introduction

Bootstrapping is a powerful resampling technique for estimating the uncertainty of statistics. It allows us to construct confidence intervals without assumptions about the underlying distribution.

## What is Bootstrapping?

```python
import numpy as np
from scipy import stats

np.random.seed(42)

print("=== BOOTSTRAPPING ===")
print("""
The Bootstrap Method:
  1. Take original sample of size n
  2. Resample WITH REPLACEMENT n times
  3. Calculate statistic of interest
  4. Repeat steps 2-3 many times (1000+)
  5. Use distribution of statistics for inference

Key Insight:
  The sample is our best estimate of the population.
  Resampling from it simulates sampling from the population.

    Population (unknown)
          ↓
    Original Sample (n observations)
          ↓
    Bootstrap Samples (resample with replacement)
          ↓
    Bootstrap Statistics (mean, median, etc.)
          ↓
    Estimate uncertainty!
""")
```

## Basic Bootstrap Example

```python
print("\n=== BASIC BOOTSTRAP ===")

# Original sample
original_sample = np.array([12, 15, 18, 22, 25, 28, 32, 35, 40, 45])
print(f"Original sample (n=10): {original_sample}")
print(f"Sample mean: {np.mean(original_sample):.2f}")

# Bootstrap
n_bootstrap = 10000
bootstrap_means = []

for _ in range(n_bootstrap):
    # Resample with replacement
    bootstrap_sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

bootstrap_means = np.array(bootstrap_means)

print(f"\nBootstrap Results ({n_bootstrap} iterations):")
print(f"  Mean of bootstrap means: {np.mean(bootstrap_means):.2f}")
print(f"  Std of bootstrap means: {np.std(bootstrap_means):.2f}")
print(f"  This std is the 'standard error' of the mean!")

# Show some bootstrap samples
print("\nFirst 5 bootstrap samples:")
np.random.seed(42)
for i in range(5):
    bs = np.random.choice(original_sample, size=len(original_sample), replace=True)
    print(f"  Bootstrap {i+1}: {bs}, mean={np.mean(bs):.2f}")
```

## Confidence Intervals

```python
print("\n=== CONFIDENCE INTERVALS ===")
print("""
A 95% Confidence Interval means:
  If we repeated the sampling process many times,
  95% of the intervals would contain the true parameter.

NOT: "95% probability the true value is in this interval"
(The true value is fixed; the interval is random)

Methods to construct CI:

1. PERCENTILE METHOD (simple)
   CI = [2.5th percentile, 97.5th percentile] of bootstrap distribution

2. STANDARD ERROR METHOD (assumes normality)
   CI = statistic ± z* × SE(bootstrap)
   
3. BCa (Bias-Corrected and Accelerated)
   Adjusts for bias and skewness
   More accurate but more complex
""")

# 95% CI using percentile method
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"\n95% Confidence Interval for the Mean:")
print(f"  Percentile method: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Using standard error
se = np.std(bootstrap_means)
mean = np.mean(original_sample)
ci_lower_se = mean - 1.96 * se
ci_upper_se = mean + 1.96 * se
print(f"  SE method: [{ci_lower_se:.2f}, {ci_upper_se:.2f}]")

# Compare with traditional t-interval
t_ci = stats.t.interval(0.95, df=len(original_sample)-1, 
                        loc=np.mean(original_sample),
                        scale=stats.sem(original_sample))
print(f"  Traditional t-interval: [{t_ci[0]:.2f}, {t_ci[1]:.2f}]")
```

## Bootstrap for Any Statistic

```python
print("\n=== BOOTSTRAP FOR ANY STATISTIC ===")

# Sample data (income-like, skewed)
np.random.seed(42)
incomes = np.random.exponential(50000, 100)
incomes = np.clip(incomes, 10000, 500000)

print(f"Income sample (n=100):")
print(f"  Mean: ${np.mean(incomes):,.0f}")
print(f"  Median: ${np.median(incomes):,.0f}")
print(f"  Std: ${np.std(incomes):,.0f}")

def bootstrap_ci(data, statistic_func, n_bootstrap=10000, ci_level=0.95):
    """Calculate bootstrap confidence interval for any statistic."""
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(bootstrap_stats, alpha * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return ci_lower, ci_upper, bootstrap_stats

# CI for mean
mean_lower, mean_upper, _ = bootstrap_ci(incomes, np.mean)
print(f"\n95% CI for Mean: [${mean_lower:,.0f}, ${mean_upper:,.0f}]")

# CI for median (hard to do analytically!)
median_lower, median_upper, _ = bootstrap_ci(incomes, np.median)
print(f"95% CI for Median: [${median_lower:,.0f}, ${median_upper:,.0f}]")

# CI for 90th percentile
def p90(x): return np.percentile(x, 90)
p90_lower, p90_upper, _ = bootstrap_ci(incomes, p90)
print(f"95% CI for 90th Percentile: [${p90_lower:,.0f}, ${p90_upper:,.0f}]")

# CI for standard deviation
std_lower, std_upper, _ = bootstrap_ci(incomes, np.std)
print(f"95% CI for Std Dev: [${std_lower:,.0f}, ${std_upper:,.0f}]")
```

## Bootstrap for Correlation

```python
print("\n=== BOOTSTRAP FOR CORRELATION ===")

# Two correlated variables
np.random.seed(42)
n = 50
x = np.random.normal(100, 15, n)
y = 0.7*x + np.random.normal(0, 10, n)

observed_corr = np.corrcoef(x, y)[0, 1]
print(f"Observed correlation: r = {observed_corr:.3f}")

# Bootstrap CI for correlation
def correlation(data):
    return np.corrcoef(data[:, 0], data[:, 1])[0, 1]

# Stack data for resampling pairs together
data = np.column_stack([x, y])

n_bootstrap = 10000
bootstrap_corrs = []

for _ in range(n_bootstrap):
    indices = np.random.choice(n, size=n, replace=True)
    sample = data[indices]
    bootstrap_corrs.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])

bootstrap_corrs = np.array(bootstrap_corrs)
ci_lower = np.percentile(bootstrap_corrs, 2.5)
ci_upper = np.percentile(bootstrap_corrs, 97.5)

print(f"95% CI for correlation: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"Bootstrap SE: {np.std(bootstrap_corrs):.3f}")
```

## Bootstrap Hypothesis Testing

```python
print("\n=== BOOTSTRAP HYPOTHESIS TEST ===")
print("""
Testing H₀: mean = μ₀ using bootstrap

Approach: 
  1. Shift data to have mean μ₀ (under H₀)
  2. Bootstrap from shifted data
  3. See how often bootstrap mean is as extreme as observed
""")

# Test H₀: μ = 25 for our original sample
original_sample = np.array([12, 15, 18, 22, 25, 28, 32, 35, 40, 45])
observed_mean = np.mean(original_sample)
null_mean = 25

print(f"Observed mean: {observed_mean}")
print(f"Null hypothesis: μ = {null_mean}")
print(f"Test statistic: {observed_mean - null_mean}")

# Shift data to satisfy H₀
shifted_sample = original_sample - (observed_mean - null_mean)

# Bootstrap under H₀
n_bootstrap = 10000
bootstrap_means_h0 = []

for _ in range(n_bootstrap):
    bs = np.random.choice(shifted_sample, size=len(shifted_sample), replace=True)
    bootstrap_means_h0.append(np.mean(bs))

bootstrap_means_h0 = np.array(bootstrap_means_h0)

# p-value (two-tailed)
extreme = np.abs(bootstrap_means_h0 - null_mean) >= np.abs(observed_mean - null_mean)
p_value = np.mean(extreme)

print(f"\nBootstrap p-value (two-tailed): {p_value:.4f}")
if p_value < 0.05:
    print("Reject H₀ at α=0.05")
else:
    print("Fail to reject H₀ at α=0.05")
```

## When to Use Bootstrap

```python
print("\n=== WHEN TO USE BOOTSTRAP ===")
print("""
ADVANTAGES:
  ✓ Works for any statistic (mean, median, correlation, etc.)
  ✓ No distributional assumptions needed
  ✓ Simple to understand and implement
  ✓ Works well for small samples
  ✓ Handles complex statistics (ratios, percentiles)

LIMITATIONS:
  ✗ Computationally intensive (need many iterations)
  ✗ Results vary slightly each run (use many iterations)
  ✗ Assumes sample is representative
  ✗ Can fail for very small samples (n < 10-15)
  ✗ Extreme percentiles may be unstable

WHEN TO USE:
  - Complex statistics without formulas
  - Non-normal data
  - Confidence intervals for median, IQR, etc.
  - When classical assumptions are violated
  - Exploratory analysis

WHEN NOT TO USE:
  - Very small samples (n < 10)
  - When classical methods work well
  - Time series with dependence (need block bootstrap)
""")
```

## Key Points

- **Bootstrap**: Resample with replacement to estimate uncertainty
- **Confidence interval**: Range likely to contain true parameter
- **Percentile method**: Simple CI from bootstrap distribution
- **Works for any statistic**: Mean, median, correlation, percentiles
- **No assumptions**: Distribution-free inference
- **Use many iterations**: 10,000+ for stable estimates
- **95% CI interpretation**: 95% of intervals capture true value (frequentist)

## Reflection Questions

1. Why do we resample with replacement in bootstrapping?
2. How is the bootstrap standard error related to the standard error of the mean?
3. When might the bootstrap give misleading results?
