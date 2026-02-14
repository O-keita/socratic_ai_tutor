# Hypothesis Testing

## Introduction

Hypothesis testing is a systematic method for making decisions about populations based on sample data. It provides a framework for determining whether observed differences are statistically significant or could have occurred by chance.

## The Logic of Hypothesis Testing

```python
import numpy as np
from scipy import stats

np.random.seed(42)

print("=== HYPOTHESIS TESTING FRAMEWORK ===")
print("""
The Process:
  1. State the hypotheses (H₀ and H₁)
  2. Choose significance level (α)
  3. Collect data and calculate test statistic
  4. Find p-value or critical value
  5. Make decision: Reject or fail to reject H₀

Key Terms:
  H₀ (Null Hypothesis): Default assumption (no effect, no difference)
  H₁ (Alternative Hypothesis): What we're trying to show
  α (Alpha): Significance level (typically 0.05)
  p-value: Probability of getting results as extreme as observed, IF H₀ is true
  
Decision Rule:
  If p-value < α: Reject H₀ (result is "statistically significant")
  If p-value ≥ α: Fail to reject H₀ (insufficient evidence)
""")
```

## One-Sample t-Test

```python
print("\n=== ONE-SAMPLE T-TEST ===")
print("""
Tests whether sample mean differs from a known value.

Example: A factory claims average widget weight is 100g.
         We sample 30 widgets. Is the claim accurate?
""")

# Sample data
np.random.seed(42)
sample = np.random.normal(102, 5, 30)  # Actually slightly above 100
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)

print(f"Sample: n={n}, mean={sample_mean:.2f}g, std={sample_std:.2f}g")
print(f"Claimed population mean: 100g")

# Hypotheses
print("\nHypotheses:")
print("  H₀: μ = 100 (claim is true)")
print("  H₁: μ ≠ 100 (claim is false)")

# Calculate test statistic
t_stat = (sample_mean - 100) / (sample_std / np.sqrt(n))
print(f"\nTest statistic:")
print(f"  t = (x̄ - μ₀) / (s/√n)")
print(f"  t = ({sample_mean:.2f} - 100) / ({sample_std:.2f}/√{n})")
print(f"  t = {t_stat:.4f}")

# p-value (two-tailed)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
print(f"\np-value (two-tailed): {p_value:.4f}")

# Decision
alpha = 0.05
print(f"\nDecision (α = {alpha}):")
if p_value < alpha:
    print(f"  p-value ({p_value:.4f}) < α ({alpha})")
    print(f"  REJECT H₀ - Evidence suggests mean ≠ 100g")
else:
    print(f"  p-value ({p_value:.4f}) ≥ α ({alpha})")
    print(f"  FAIL TO REJECT H₀ - Insufficient evidence")

# Using scipy directly
t_stat_scipy, p_value_scipy = stats.ttest_1samp(sample, 100)
print(f"\nUsing scipy.stats.ttest_1samp:")
print(f"  t = {t_stat_scipy:.4f}, p = {p_value_scipy:.4f}")
```

## Two-Sample t-Test

```python
print("\n=== TWO-SAMPLE T-TEST ===")
print("""
Tests whether two independent groups have different means.

Example: Does a new teaching method improve test scores?
""")

# Two groups
np.random.seed(42)
control = np.random.normal(75, 10, 35)   # Traditional method
treatment = np.random.normal(80, 10, 35)  # New method

print(f"Control group: n={len(control)}, mean={np.mean(control):.2f}")
print(f"Treatment group: n={len(treatment)}, mean={np.mean(treatment):.2f}")

# Hypotheses
print("\nHypotheses:")
print("  H₀: μ₁ = μ₂ (no difference)")
print("  H₁: μ₁ ≠ μ₂ (there is a difference)")

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f"\nTest results:")
print(f"  t = {t_stat:.4f}")
print(f"  p-value = {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"\nConclusion: Reject H₀ (p < {alpha})")
    print("  The new teaching method appears to have an effect!")
else:
    print(f"\nConclusion: Fail to reject H₀ (p ≥ {alpha})")
```

## One-Tailed vs Two-Tailed Tests

```python
print("\n=== ONE-TAILED VS TWO-TAILED ===")
print("""
TWO-TAILED (most common):
  H₁: μ ≠ μ₀ (different in either direction)
  Rejects if too high OR too low
  
ONE-TAILED (directional):
  H₁: μ > μ₀ (greater than) or H₁: μ < μ₀ (less than)
  More powerful but only detects difference in one direction

Example: Testing if a drug increases blood pressure
  Two-tailed: H₁: μ ≠ 120 (any change)
  One-tailed: H₁: μ > 120 (increase only)
""")

# Example
np.random.seed(42)
sample = np.random.normal(125, 10, 25)
null_value = 120

# Two-tailed
t_stat, p_two = stats.ttest_1samp(sample, null_value)
print(f"Sample mean: {np.mean(sample):.2f}")
print(f"Testing against μ₀ = {null_value}")
print(f"\nTwo-tailed test: H₁: μ ≠ {null_value}")
print(f"  p-value = {p_two:.4f}")

# One-tailed (greater than)
p_one = p_two / 2 if t_stat > 0 else 1 - p_two/2
print(f"\nOne-tailed test: H₁: μ > {null_value}")
print(f"  p-value = {p_one:.4f}")
```

## Type I and Type II Errors

```python
print("\n=== ERRORS IN HYPOTHESIS TESTING ===")
print("""
                           Reality
                    H₀ True    H₀ False
Decision  ─────────────────────────────
Reject H₀    Type I Error    Correct!
             (False Positive) (True Positive)
             
Fail to      Correct!        Type II Error
Reject H₀    (True Negative)  (False Negative)

TYPE I ERROR (α):
  - Rejecting H₀ when it's actually true
  - "False alarm" / "False positive"
  - Probability = α (significance level)
  - Example: Concluding drug works when it doesn't

TYPE II ERROR (β):
  - Failing to reject H₀ when it's actually false  
  - "Missed detection" / "False negative"
  - Probability = β
  - Example: Missing that drug actually works

POWER = 1 - β
  - Probability of correctly rejecting false H₀
  - Higher power = better at detecting real effects
  - Increase power by: larger n, larger effect, higher α
""")

# Demonstrate with simulation
def simulate_errors(n_simulations=10000, effect_size=0, n=30, alpha=0.05):
    """Simulate hypothesis tests."""
    type1_count = 0
    type2_count = 0
    
    for _ in range(n_simulations):
        # Generate sample (effect_size = 0 means H0 is true)
        sample = np.random.normal(100 + effect_size, 15, n)
        _, p_value = stats.ttest_1samp(sample, 100)
        
        if effect_size == 0:  # H0 is true
            if p_value < alpha:
                type1_count += 1  # False positive
        else:  # H0 is false
            if p_value >= alpha:
                type2_count += 1  # False negative
    
    return type1_count / n_simulations, type2_count / n_simulations

# When H0 is true (no effect)
type1_rate, _ = simulate_errors(effect_size=0)
print(f"\nSimulation (H₀ true, no effect):")
print(f"  Type I error rate: {type1_rate:.3f} (expected ≈ 0.05)")

# When H0 is false (real effect)
_, type2_rate = simulate_errors(effect_size=5)
print(f"\nSimulation (H₀ false, effect=5):")
print(f"  Type II error rate: {type2_rate:.3f}")
print(f"  Power: {1-type2_rate:.3f}")
```

## Common Statistical Tests

```python
print("\n=== COMMON STATISTICAL TESTS ===")
print("""
COMPARING MEANS:
  - One-sample t-test: Sample mean vs known value
  - Two-sample t-test: Compare two independent groups
  - Paired t-test: Compare matched pairs (before/after)
  
COMPARING PROPORTIONS:
  - One-proportion z-test: Sample proportion vs known value
  - Two-proportion z-test: Compare two groups
  - Chi-square test: Independence of categorical variables
  
NON-PARAMETRIC (when assumptions violated):
  - Mann-Whitney U: Compare two groups (non-normal)
  - Wilcoxon signed-rank: Paired comparison (non-normal)
  - Kruskal-Wallis: Compare 3+ groups (non-normal)
  
MORE THAN TWO GROUPS:
  - ANOVA: Compare means of 3+ groups
  - Post-hoc tests: Determine which groups differ
""")

# Chi-square example
print("\nChi-Square Test Example:")
print("Is there a relationship between gender and product preference?")

# Observed frequencies (contingency table)
observed = np.array([[30, 10, 20],   # Male: A, B, C
                     [20, 30, 10]])  # Female: A, B, C

chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"\nObserved frequencies:\n{observed}")
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
if p_value < 0.05:
    print("Conclusion: Significant relationship exists!")
```

## Key Points

- **Null hypothesis (H₀)**: Default assumption of no effect
- **Alternative hypothesis (H₁)**: What we're trying to demonstrate
- **p-value**: Probability of results if H₀ true; low p suggests H₀ unlikely
- **α (alpha)**: Threshold for "statistical significance" (usually 0.05)
- **Type I error**: False positive (rejecting true H₀)
- **Type II error**: False negative (missing real effect)
- **Power**: Probability of detecting real effect (1 - β)
- **Statistical significance ≠ practical significance**

## Reflection Questions

1. Why do we "fail to reject" rather than "accept" the null hypothesis?
2. How does sample size affect the power of a test?
3. When might a statistically significant result not be practically meaningful?
