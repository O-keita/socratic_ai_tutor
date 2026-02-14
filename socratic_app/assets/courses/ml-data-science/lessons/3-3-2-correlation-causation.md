# Correlation vs Causation

## Introduction

Understanding the difference between correlation and causation is one of the most important concepts in data science. Just because two variables are related doesn't mean one causes the other.

## Correlation Basics

```python
import numpy as np
from scipy import stats

np.random.seed(42)

print("=== CORRELATION ===")
print("""
Correlation measures the LINEAR relationship between two variables.

Pearson Correlation Coefficient (r):
  - Ranges from -1 to +1
  - r = +1: Perfect positive linear relationship
  - r = -1: Perfect negative linear relationship
  - r = 0: No linear relationship
  
Interpretation:
  |r| < 0.3: Weak
  0.3 ≤ |r| < 0.7: Moderate
  |r| ≥ 0.7: Strong
""")

# Generate correlated data
n = 100
x = np.random.normal(50, 10, n)
y_positive = 2*x + np.random.normal(0, 10, n)  # Positive correlation
y_negative = -1.5*x + np.random.normal(0, 10, n)  # Negative correlation
y_none = np.random.normal(50, 15, n)  # No correlation

print("Correlation examples:")
print(f"  Positive correlation: r = {np.corrcoef(x, y_positive)[0,1]:.3f}")
print(f"  Negative correlation: r = {np.corrcoef(x, y_negative)[0,1]:.3f}")
print(f"  No correlation: r = {np.corrcoef(x, y_none)[0,1]:.3f}")
```

## Correlation Does NOT Imply Causation

```python
print("\n=== CORRELATION ≠ CAUSATION ===")
print("""
Just because X and Y are correlated does NOT mean:
  - X causes Y
  - Y causes X
  - Either causes the other

Famous Examples:

1. ICE CREAM SALES & DROWNING DEATHS
   Correlation: Positive (both increase in summer)
   Causation: NO! Both caused by hot weather (confounding variable)

2. SHOE SIZE & READING ABILITY (in children)
   Correlation: Positive
   Causation: NO! Both increase with age

3. PIRATES & GLOBAL WARMING
   Correlation: Negative (fewer pirates = warmer planet)
   Causation: NO! Just coincidental trends

4. NICHOLAS CAGE MOVIES & POOL DROWNINGS
   Correlation: r ≈ 0.67
   Causation: Obviously not! Spurious correlation
""")
```

## Why Correlation ≠ Causation

```python
print("\n=== REASONS CORRELATION ≠ CAUSATION ===")
print("""
1. CONFOUNDING VARIABLES (Third Variable Problem)
   
   Observed: A ←→ B (correlated)
   Reality:  A ← C → B (C causes both)
   
   Example: Coffee drinking ↔ Lung cancer
   Confounder: Smoking (smokers drink more coffee AND get cancer)

2. REVERSE CAUSATION
   
   Observed: A → B (we think A causes B)
   Reality:  A ← B (B actually causes A)
   
   Example: Do firefighters cause fires?
   (More firefighters at bigger fires)

3. COINCIDENCE / SPURIOUS CORRELATION
   
   No real relationship, just random chance
   Example: Cheese consumption vs PhD graduates

4. SELECTION BIAS
   
   Only seeing part of the data creates false patterns
   Example: Hospital data showing treatment = worse outcomes
   (Sicker people get more treatment)

5. MEASUREMENT ERROR
   
   Poor measurement can create or hide relationships
""")

# Demonstrate confounding
print("\nDemonstration: Confounding Variable")
print("-" * 40)

np.random.seed(42)
n = 200

# C is the confounder
C = np.random.normal(0, 1, n)

# A and B are both caused by C (not by each other)
A = 2*C + np.random.normal(0, 0.5, n)
B = 3*C + np.random.normal(0, 0.5, n)

print(f"Correlation(A, B) = {np.corrcoef(A, B)[0,1]:.3f}")
print("High correlation! But A and B don't cause each other.")
print("Both are caused by confounding variable C.")
```

## How to Establish Causation

```python
print("\n=== ESTABLISHING CAUSATION ===")
print("""
To claim X causes Y, you typically need:

1. RANDOMIZED CONTROLLED EXPERIMENTS (Gold Standard)
   - Randomly assign subjects to treatment/control
   - Control for confounders
   - Example: Clinical drug trials

2. NATURAL EXPERIMENTS
   - "Random" assignment happens naturally
   - Example: Twin studies, policy changes

3. QUASI-EXPERIMENTS
   - Compare before/after
   - Regression discontinuity
   - Difference-in-differences

4. BRADFORD HILL CRITERIA (for observational studies)
   - Strength of association
   - Consistency across studies
   - Specificity
   - Temporality (cause precedes effect)
   - Biological gradient (dose-response)
   - Plausibility
   - Coherence
   - Experiment (if possible)
   - Analogy

Key principle: TEMPORALITY
  - Cause must come BEFORE effect
  - Necessary but not sufficient
""")
```

## Simpson's Paradox

```python
print("\n=== SIMPSON'S PARADOX ===")
print("""
A trend in grouped data can REVERSE when groups are combined!

Example: Treatment Success Rates

Hospital A:
  Treatment: 93/100 survived (93%)
  No treatment: 87/100 survived (87%)
  → Treatment seems better!

Hospital B:
  Treatment: 7/10 survived (70%)
  No treatment: 3/10 survived (30%)
  → Treatment seems better!

Combined:
  Treatment: 100/110 survived (91%)
  No treatment: 90/110 survived (82%)
  → Treatment still better? Let's check severity...
""")

# Numerical demonstration
print("\nActual data by patient severity:")
print("""
MILD CASES:
  Treatment: 90/100 survived (90%)
  No treatment: 85/90 survived (94%)
  → No treatment is better for mild cases!

SEVERE CASES:
  Treatment: 10/10 survived (100%)
  No treatment: 5/20 survived (25%)
  → Treatment is better for severe cases!

The catch: Hospital A sees mostly mild cases (prefers no treatment)
          Hospital B sees mostly severe cases (needs treatment)

Simpson's Paradox: Must consider confounding variable (severity)!
""")
```

## Correlation Matrix Analysis

```python
print("\n=== CORRELATION MATRIX ===")

# Create sample data
np.random.seed(42)
n = 100

data = {
    'study_hours': np.random.uniform(0, 10, n),
    'sleep_hours': np.random.uniform(5, 9, n),
    'coffee_cups': np.random.uniform(0, 5, n)
}
# Test score depends on study and sleep, not coffee
data['test_score'] = (
    60 + 
    3 * data['study_hours'] + 
    2 * data['sleep_hours'] + 
    np.random.normal(0, 5, n)
)
# But coffee correlates with study (students drink coffee while studying)
data['coffee_cups'] = 0.3 * data['study_hours'] + np.random.normal(2, 0.5, n)

import pandas as pd
df = pd.DataFrame(data)

print("Correlation Matrix:")
print(df.corr().round(3))

print("""
Observations:
  - study_hours ↔ test_score: Strong (r≈0.72) - CAUSAL
  - sleep_hours ↔ test_score: Moderate (r≈0.33) - CAUSAL
  - coffee_cups ↔ test_score: Moderate (r≈0.38) - CONFOUNDED!
    (Coffee doesn't help scores; it's correlated with studying)
""")
```

## Practical Guidelines

```python
print("\n=== PRACTICAL GUIDELINES ===")
print("""
When you find a correlation:

1. ASK: Does X cause Y, Y cause X, or neither?

2. LOOK FOR CONFOUNDERS
   - What else could explain this relationship?
   - Are there hidden variables?

3. CHECK TEMPORALITY
   - Does the "cause" come before the "effect"?

4. CONSIDER THE MECHANISM
   - Is there a plausible explanation?
   - What's the theoretical basis?

5. LOOK FOR DOSE-RESPONSE
   - More X → more Y (in consistent manner)?

6. CHECK OTHER STUDIES
   - Has this been replicated?
   - Are there experiments?

7. BE SKEPTICAL
   - "Correlation is not causation" should be your mantra
   - Extraordinary claims require extraordinary evidence

LANGUAGE MATTERS:
  ✗ "X causes Y" (unless proven)
  ✗ "X leads to Y"
  ✓ "X is associated with Y"
  ✓ "X is correlated with Y"
  ✓ "X predicts Y" (for ML, not causation)
""")
```

## Key Points

- **Correlation**: Measures linear relationship (-1 to +1)
- **Causation**: X directly produces change in Y
- **Confounding**: Third variable causes both X and Y
- **Reverse causation**: Direction might be opposite
- **Simpson's Paradox**: Trends can reverse when grouping changes
- **Experiments**: Best way to establish causation
- **Language**: Say "associated with" not "causes" unless proven

## Reflection Questions

1. How would you test if a correlation represents causation?
2. What confounding variables might explain the correlation between education and income?
3. Why are randomized controlled trials the gold standard for establishing causation?
