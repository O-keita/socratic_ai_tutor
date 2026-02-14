# Normal Distributions

## Introduction

The normal distribution (Gaussian distribution) is the most important probability distribution in statistics. It describes many natural phenomena and forms the foundation for statistical inference.

## The Normal Distribution

```python
import numpy as np
from scipy import stats

np.random.seed(42)

print("=== THE NORMAL DISTRIBUTION ===")
print("""
Also called: Gaussian distribution, bell curve

Probability Density Function:
  f(x) = (1 / σ√2π) × exp(-(x-μ)²/2σ²)

Parameters:
  μ (mu) = mean (center of distribution)
  σ (sigma) = standard deviation (spread)
  
Properties:
  - Symmetric around μ
  - Mean = Median = Mode
  - Defined by just two parameters
  - Bell-shaped curve
  - Total area under curve = 1

Notation: X ~ N(μ, σ²)
""")

# Generate and describe normal data
normal_data = np.random.normal(loc=100, scale=15, size=10000)
print(f"Sample from N(100, 15²):")
print(f"  Mean: {np.mean(normal_data):.2f}")
print(f"  Std Dev: {np.std(normal_data):.2f}")
print(f"  Min: {np.min(normal_data):.2f}")
print(f"  Max: {np.max(normal_data):.2f}")
```

## The Empirical Rule (68-95-99.7)

```python
print("\n=== EMPIRICAL RULE ===")
print("""
For any normal distribution:

  Within ±1σ of μ: ~68.27% of data
  Within ±2σ of μ: ~95.45% of data  
  Within ±3σ of μ: ~99.73% of data

Visual:
                   99.7%
            ◄─────────────────►
                  95%
             ◄───────────────►
                68%
              ◄───────────►
              ┌───────────┐
           ▄▄▄█           █▄▄▄
         ▄▄███             ███▄▄
       ▄▄█████             █████▄▄
    ───┼───┼───┼───┼───┼───┼───┼───
     -3σ -2σ -1σ  μ  1σ  2σ  3σ
""")

# Verify with simulation
mu, sigma = 100, 15
data = np.random.normal(mu, sigma, 100000)

within_1sd = np.mean((data >= mu-sigma) & (data <= mu+sigma)) * 100
within_2sd = np.mean((data >= mu-2*sigma) & (data <= mu+2*sigma)) * 100
within_3sd = np.mean((data >= mu-3*sigma) & (data <= mu+3*sigma)) * 100

print(f"Simulation verification (N={len(data):,}):")
print(f"  Within ±1σ: {within_1sd:.2f}% (theory: 68.27%)")
print(f"  Within ±2σ: {within_2sd:.2f}% (theory: 95.45%)")
print(f"  Within ±3σ: {within_3sd:.2f}% (theory: 99.73%)")
```

## The Standard Normal Distribution

```python
print("\n=== STANDARD NORMAL DISTRIBUTION ===")
print("""
The Standard Normal has:
  μ = 0
  σ = 1

Notation: Z ~ N(0, 1)

Z-SCORE (Standardization):
  Z = (X - μ) / σ
  
  "How many standard deviations from the mean"
  
Converting back:
  X = μ + Z × σ
""")

# Z-score example
exam_mean = 75
exam_std = 10
student_score = 90

z_score = (student_score - exam_mean) / exam_std
print(f"\nExample: Exam scores ~ N({exam_mean}, {exam_std}²)")
print(f"Student scored: {student_score}")
print(f"Z-score: ({student_score} - {exam_mean}) / {exam_std} = {z_score}")
print(f"Interpretation: {z_score} standard deviations above the mean")

# What percentile?
percentile = stats.norm.cdf(z_score) * 100
print(f"Percentile: {percentile:.1f}th (better than {percentile:.1f}% of students)")
```

## Calculating Probabilities

```python
print("\n=== CALCULATING PROBABILITIES ===")

# Using scipy.stats
mu, sigma = 100, 15

print("Distribution: X ~ N(100, 15²)")

# P(X < 120)
p_less_120 = stats.norm.cdf(120, loc=mu, scale=sigma)
print(f"\nP(X < 120) = {p_less_120:.4f}")

# P(X > 130)
p_greater_130 = 1 - stats.norm.cdf(130, loc=mu, scale=sigma)
print(f"P(X > 130) = {p_greater_130:.4f}")

# P(85 < X < 115)
p_between = stats.norm.cdf(115, loc=mu, scale=sigma) - stats.norm.cdf(85, loc=mu, scale=sigma)
print(f"P(85 < X < 115) = {p_between:.4f}")

# Using Z-scores
print("\nUsing Z-scores (standard normal):")
z_120 = (120 - mu) / sigma
p_less_120_z = stats.norm.cdf(z_120)
print(f"P(X < 120): Z = {z_120:.2f}, P(Z < {z_120:.2f}) = {p_less_120_z:.4f}")
```

## Finding Values from Probabilities

```python
print("\n=== INVERSE: FINDING VALUES ===")

mu, sigma = 100, 15

# What score is the 90th percentile?
p90 = stats.norm.ppf(0.90, loc=mu, scale=sigma)
print(f"90th percentile: {p90:.2f}")

# What scores bound the middle 95%?
lower = stats.norm.ppf(0.025, loc=mu, scale=sigma)
upper = stats.norm.ppf(0.975, loc=mu, scale=sigma)
print(f"Middle 95%: [{lower:.2f}, {upper:.2f}]")

# Top 5%
top_5 = stats.norm.ppf(0.95, loc=mu, scale=sigma)
print(f"Top 5% starts at: {top_5:.2f}")

# Using Z-scores
z_90 = stats.norm.ppf(0.90)  # Standard normal
x_90 = mu + z_90 * sigma     # Convert to X
print(f"\nUsing Z: z₉₀ = {z_90:.4f}, x₉₀ = {mu} + {z_90:.4f}×{sigma} = {x_90:.2f}")
```

## Common Z-Score Reference Values

```python
print("\n=== COMMON Z-SCORE VALUES ===")
print("""
Cumulative Probability P(Z < z):
  
  z = -2.00  →  P = 0.0228 (2.3%)
  z = -1.96  →  P = 0.0250 (2.5%)  ← Used for 95% CI
  z = -1.00  →  P = 0.1587 (15.9%)
  z =  0.00  →  P = 0.5000 (50%)
  z =  1.00  →  P = 0.8413 (84.1%)
  z =  1.64  →  P = 0.9495 (95%)  ← Used for 90% CI
  z =  1.96  →  P = 0.9750 (97.5%) ← Used for 95% CI
  z =  2.00  →  P = 0.9772 (97.7%)
  z =  2.58  →  P = 0.9951 (99.5%) ← Used for 99% CI

Common confidence intervals:
  90% CI: z = ±1.645
  95% CI: z = ±1.960
  99% CI: z = ±2.576
""")

# Verify
for z in [-1.96, -1, 0, 1, 1.96]:
    p = stats.norm.cdf(z)
    print(f"P(Z < {z:5.2f}) = {p:.4f}")
```

## Why is the Normal Distribution Important?

```python
print("\n=== WHY NORMAL MATTERS ===")
print("""
1. NATURAL PHENOMENA
   Many real-world measurements are approximately normal:
   - Human heights, weights
   - Measurement errors
   - IQ scores
   - Blood pressure
   
2. CENTRAL LIMIT THEOREM (CLT)
   Sample means approach normal distribution regardless
   of the original distribution's shape!
   
   X̄ ~ N(μ, σ²/n) as n gets large
   
3. STATISTICAL INFERENCE
   - Confidence intervals
   - Hypothesis tests
   - Regression analysis
   
4. MAXIMUM ENTROPY
   Among distributions with given mean and variance,
   normal has maximum entropy (most uncertain)
""")

# Demonstrate CLT
np.random.seed(42)

# Original distribution: Uniform (not normal!)
uniform_data = np.random.uniform(0, 10, 100000)
print(f"Uniform distribution: mean={np.mean(uniform_data):.2f}, std={np.std(uniform_data):.2f}")

# Sample means
sample_sizes = [1, 2, 5, 30]
for n in sample_sizes:
    samples = uniform_data.reshape(-1, n)[:1000]  # 1000 samples of size n
    sample_means = np.mean(samples, axis=1)
    print(f"  Sample means (n={n:2d}): mean={np.mean(sample_means):.2f}, std={np.std(sample_means):.2f}")

print("\nAs n increases, sample means become more normal!")
```

## Checking Normality

```python
print("\n=== CHECKING NORMALITY ===")
print("""
Methods to check if data is approximately normal:

1. VISUAL: Histogram, Q-Q plot
2. SKEWNESS & KURTOSIS
   - Skewness ≈ 0
   - Kurtosis ≈ 0 (excess)
3. STATISTICAL TESTS
   - Shapiro-Wilk test
   - Kolmogorov-Smirnov test
""")

# Generate normal and non-normal data
normal = np.random.normal(0, 1, 1000)
skewed = np.random.exponential(1, 1000)

# Shapiro-Wilk test
stat_normal, p_normal = stats.shapiro(normal[:500])  # Use subset for shapiro
stat_skewed, p_skewed = stats.shapiro(skewed[:500])

print(f"Shapiro-Wilk Test (H₀: data is normal):")
print(f"  Normal data: statistic={stat_normal:.4f}, p-value={p_normal:.4f}")
print(f"  Skewed data: statistic={stat_skewed:.4f}, p-value={p_skewed:.4f}")
print(f"\nIf p < 0.05, reject normality assumption")
```

## Key Points

- **Normal distribution**: Bell-shaped, symmetric, defined by μ and σ
- **Standard normal**: N(0,1), Z-scores measure deviations from mean
- **Empirical rule**: 68-95-99.7% within ±1, ±2, ±3 standard deviations
- **Z-score**: Z = (X - μ) / σ standardizes any normal distribution
- **scipy.stats.norm**: `cdf()` for probabilities, `ppf()` for inverse
- **Central Limit Theorem**: Sample means are approximately normal
- **Check normality**: Q-Q plots, Shapiro-Wilk test

## Reflection Questions

1. Why is the standard normal distribution useful for any normal distribution?
2. What does the Central Limit Theorem allow us to do in practice?
3. When might data appear normal but actually come from a different distribution?
