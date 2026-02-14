# Modeling Chance Events

## Introduction

Probability is the mathematical framework for modeling uncertainty and chance. Understanding probability is essential for statistical inference, machine learning, and making decisions under uncertainty.

## Basic Probability Concepts

```python
import numpy as np
from itertools import combinations, product

np.random.seed(42)

print("=== BASIC PROBABILITY ===")
print("""
PROBABILITY FUNDAMENTALS:

Sample Space (S): All possible outcomes
Event (E): A subset of outcomes we're interested in
Probability: P(E) = Number of favorable outcomes / Total outcomes

Properties:
  1. 0 ≤ P(E) ≤ 1
  2. P(S) = 1 (something must happen)
  3. P(not E) = 1 - P(E)
  
Interpretations:
  - Frequentist: Long-run relative frequency
  - Bayesian: Degree of belief
""")

# Example: Rolling a fair die
print("\nExample: Fair 6-sided die")
print("Sample space S = {1, 2, 3, 4, 5, 6}")

# P(rolling a 3)
p_three = 1 / 6
print(f"P(rolling 3) = 1/6 = {p_three:.4f}")

# P(rolling even)
p_even = 3 / 6
print(f"P(rolling even) = 3/6 = {p_even:.4f}")

# P(rolling > 4)
p_greater_4 = 2 / 6
print(f"P(rolling > 4) = 2/6 = {p_greater_4:.4f}")
```

## Combining Events: AND, OR, NOT

```python
print("\n=== COMBINING EVENTS ===")

# Two dice
print("Example: Rolling two fair dice")
total_outcomes = 36
print(f"Total outcomes: 6 × 6 = {total_outcomes}")

# Count outcomes for sum = 7
sum_7_outcomes = [(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)]
p_sum_7 = len(sum_7_outcomes) / total_outcomes
print(f"\nP(sum = 7) = {len(sum_7_outcomes)}/36 = {p_sum_7:.4f}")

print("""
ADDITION RULE (OR):
  P(A or B) = P(A) + P(B) - P(A and B)
  
  If A and B are mutually exclusive (can't both happen):
  P(A or B) = P(A) + P(B)

MULTIPLICATION RULE (AND):
  P(A and B) = P(A) × P(B|A)
  
  If A and B are independent:
  P(A and B) = P(A) × P(B)

COMPLEMENT (NOT):
  P(not A) = 1 - P(A)
""")

# Example: Cards
print("\nExample: Drawing from a standard deck")
p_heart = 13/52  # P(heart)
p_king = 4/52    # P(king)
p_heart_and_king = 1/52  # P(king of hearts)

p_heart_or_king = p_heart + p_king - p_heart_and_king
print(f"P(heart) = 13/52 = {p_heart:.4f}")
print(f"P(king) = 4/52 = {p_king:.4f}")
print(f"P(heart AND king) = 1/52 = {p_heart_and_king:.4f}")
print(f"P(heart OR king) = {p_heart} + {p_king} - {p_heart_and_king} = {p_heart_or_king:.4f}")
```

## Conditional Probability

```python
print("\n=== CONDITIONAL PROBABILITY ===")
print("""
P(A|B) = "Probability of A given B has occurred"

Formula:
  P(A|B) = P(A and B) / P(B)

Example: Medical test
  - 1% of population has disease (D)
  - Test is 99% accurate for sick people: P(+|D) = 0.99
  - Test is 95% accurate for healthy people: P(-|not D) = 0.95
""")

# Medical test example
p_disease = 0.01
p_no_disease = 0.99
p_positive_given_disease = 0.99
p_negative_given_no_disease = 0.95
p_positive_given_no_disease = 1 - p_negative_given_no_disease  # False positive

# P(positive test)
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)

print(f"P(disease) = {p_disease}")
print(f"P(positive | disease) = {p_positive_given_disease}")
print(f"P(positive | no disease) = {p_positive_given_no_disease}")
print(f"\nP(positive test) = {p_positive:.4f}")

# This leads to Bayes' theorem...
```

## Independence

```python
print("\n=== INDEPENDENCE ===")
print("""
Events A and B are INDEPENDENT if:
  P(A and B) = P(A) × P(B)
  
Equivalently:
  P(A|B) = P(A)  (knowing B doesn't change P(A))
  
Examples of independent events:
  - Consecutive coin flips
  - Rolling different dice
  - Unrelated measurements
  
Examples of dependent events:
  - Drawing cards without replacement
  - Test score and study time
  - Weather and umbrella usage
""")

# Coin flips
print("\nExample: Three coin flips")
p_heads = 0.5
p_three_heads = p_heads ** 3
print(f"P(H) = {p_heads}")
print(f"P(HHH) = {p_heads}³ = {p_three_heads}")

# Verify with simulation
np.random.seed(42)
flips = np.random.choice(['H', 'T'], size=(100000, 3))
three_heads = np.sum(np.all(flips == 'H', axis=1)) / 100000
print(f"Simulated P(HHH): {three_heads:.4f}")
```

## Random Variables

```python
print("\n=== RANDOM VARIABLES ===")
print("""
A Random Variable (RV) maps outcomes to numbers.

DISCRETE RV:
  - Finite or countably infinite values
  - Probability Mass Function (PMF): P(X = x)
  - Examples: Dice roll, number of heads
  
CONTINUOUS RV:
  - Infinite values in a range
  - Probability Density Function (PDF): f(x)
  - P(X = x) = 0 for any specific x!
  - Calculate P(a < X < b) using integrals
  - Examples: Height, time, temperature
""")

# Discrete RV example: Sum of two dice
outcomes = []
for d1 in range(1, 7):
    for d2 in range(1, 7):
        outcomes.append(d1 + d2)

# Create PMF
from collections import Counter
pmf = Counter(outcomes)
for val in sorted(pmf.keys()):
    prob = pmf[val] / 36
    print(f"P(X = {val:2d}) = {pmf[val]:2d}/36 = {prob:.4f} {'*' * int(prob * 50)}")
```

## Expected Value and Variance

```python
print("\n=== EXPECTED VALUE ===")
print("""
Expected Value E[X] = "Long-run average"

For discrete RV:
  E[X] = Σ x × P(X = x)

For continuous RV:
  E[X] = ∫ x × f(x) dx

Properties:
  - E[aX + b] = a × E[X] + b
  - E[X + Y] = E[X] + E[Y]  (always!)
""")

# Expected value of single die
x_values = np.array([1, 2, 3, 4, 5, 6])
p_values = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
expected_value = np.sum(x_values * p_values)
print(f"\nSingle die roll:")
print(f"E[X] = (1+2+3+4+5+6)/6 = {expected_value}")

# Variance
print("\n=== VARIANCE ===")
print("""
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

"Average squared deviation from the mean"

Standard Deviation: σ = √Var(X)

Properties:
  - Var(aX + b) = a² × Var(X)  (b doesn't matter!)
  - Var(X + Y) = Var(X) + Var(Y)  (if independent)
""")

# Variance of single die
e_x_squared = np.sum(x_values**2 * p_values)
variance = e_x_squared - expected_value**2
std_dev = np.sqrt(variance)
print(f"\nSingle die roll:")
print(f"E[X²] = {e_x_squared:.4f}")
print(f"Var(X) = {e_x_squared:.4f} - {expected_value}² = {variance:.4f}")
print(f"SD(X) = {std_dev:.4f}")
```

## Simulation for Probability

```python
print("\n=== MONTE CARLO SIMULATION ===")

np.random.seed(42)

# Problem: Probability that max of 3 dice is 6
def simulate_max_is_6(n_simulations):
    rolls = np.random.randint(1, 7, size=(n_simulations, 3))
    max_values = np.max(rolls, axis=1)
    return np.mean(max_values == 6)

# Exact calculation
p_at_least_one_6 = 1 - (5/6)**3
print(f"P(max of 3 dice is 6):")
print(f"  Exact: 1 - (5/6)³ = {p_at_least_one_6:.4f}")
print(f"  Simulated (10K): {simulate_max_is_6(10000):.4f}")
print(f"  Simulated (100K): {simulate_max_is_6(100000):.4f}")

# Birthday problem
def birthday_simulation(n_people, n_simulations=10000):
    matches = 0
    for _ in range(n_simulations):
        birthdays = np.random.randint(0, 365, n_people)
        if len(birthdays) != len(set(birthdays)):
            matches += 1
    return matches / n_simulations

print(f"\nBirthday Problem (P of shared birthday):")
for n in [10, 23, 30, 50]:
    print(f"  {n} people: {birthday_simulation(n):.3f}")
```

## Key Points

- **Probability**: Measure of uncertainty, 0 ≤ P ≤ 1
- **Addition rule**: P(A or B) = P(A) + P(B) - P(A and B)
- **Multiplication rule**: P(A and B) = P(A) × P(B|A)
- **Independence**: P(A and B) = P(A) × P(B)
- **Expected value**: Long-run average
- **Variance**: Spread around expected value
- **Simulation**: Approximate probabilities through repeated trials

## Reflection Questions

1. Why is P(A or B) ≠ P(A) + P(B) in general?
2. How does the birthday problem illustrate counterintuitive probability?
3. When would you use simulation instead of exact calculation?
