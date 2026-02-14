# Bayes' Theorem

## Introduction

Bayes' Theorem is a fundamental result in probability that allows us to update our beliefs based on new evidence. It forms the foundation of Bayesian statistics and has wide applications in machine learning, medical diagnosis, spam filtering, and more.

## Bayes' Theorem Formula

```python
import numpy as np

np.random.seed(42)

print("=== BAYES' THEOREM ===")
print("""
               P(B|A) × P(A)
  P(A|B) = ─────────────────────
                  P(B)

Terminology:
  P(A)   = Prior probability (belief before evidence)
  P(B|A) = Likelihood (probability of evidence given A)
  P(B)   = Marginal likelihood (total probability of evidence)
  P(A|B) = Posterior probability (updated belief after evidence)

Alternative form (useful for two hypotheses):
  
              P(B|A) × P(A)
  P(A|B) = ────────────────────────────────
           P(B|A)×P(A) + P(B|not A)×P(not A)
           
Key insight: Bayes' theorem lets us "flip" conditional probabilities!
  We know P(B|A), but we want P(A|B)
""")
```

## Classic Example: Medical Testing

```python
print("\n=== MEDICAL TESTING EXAMPLE ===")
print("""
Scenario:
  - 1% of population has a disease (prevalence)
  - Test is 99% sensitive (detects disease if present)
  - Test is 95% specific (negative if healthy)
  
Question: If you test positive, what's the probability you have the disease?
""")

# Given information
p_disease = 0.01        # P(D) - prior
p_no_disease = 0.99     # P(not D)
p_positive_given_disease = 0.99    # P(+|D) - sensitivity
p_negative_given_no_disease = 0.95 # P(-|not D) - specificity
p_positive_given_no_disease = 0.05 # P(+|not D) - false positive rate

print("Given:")
print(f"  P(Disease) = {p_disease}")
print(f"  P(+|Disease) = {p_positive_given_disease}")
print(f"  P(+|No Disease) = {p_positive_given_no_disease}")

# Calculate P(positive)
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)
print(f"\nStep 1: Calculate P(+)")
print(f"  P(+) = P(+|D)×P(D) + P(+|not D)×P(not D)")
print(f"  P(+) = {p_positive_given_disease}×{p_disease} + {p_positive_given_no_disease}×{p_no_disease}")
print(f"  P(+) = {p_positive:.4f}")

# Apply Bayes' theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
print(f"\nStep 2: Apply Bayes' theorem")
print(f"  P(D|+) = P(+|D) × P(D) / P(+)")
print(f"  P(D|+) = {p_positive_given_disease} × {p_disease} / {p_positive:.4f}")
print(f"  P(D|+) = {p_disease_given_positive:.4f} = {p_disease_given_positive*100:.1f}%")

print(f"""
Result: Only {p_disease_given_positive*100:.1f}% chance of having disease!

Why so low despite 99% test accuracy?
  - Disease is rare (1% prevalence)
  - Many more healthy people being tested
  - False positives from healthy people outnumber true positives
  
This is the "base rate fallacy" - ignoring prior probabilities!
""")
```

## Visualizing with Natural Frequencies

```python
print("\n=== NATURAL FREQUENCIES ===")
print("""
Often easier to understand with counts:

Imagine 10,000 people:
  - 100 have disease (1%)
  - 9,900 are healthy (99%)

Among 100 with disease:
  - 99 test positive (true positive)
  - 1 test negative (false negative)

Among 9,900 healthy:
  - 495 test positive (false positive)
  - 9,405 test negative (true negative)

Total positive tests: 99 + 495 = 594
True positives: 99

P(Disease | Positive) = 99 / 594 = 16.7%
""")

# Calculate with counts
population = 10000
with_disease = int(population * 0.01)
without_disease = population - with_disease

true_positives = int(with_disease * 0.99)
false_negatives = with_disease - true_positives
false_positives = int(without_disease * 0.05)
true_negatives = without_disease - false_positives

total_positive = true_positives + false_positives

print("Confusion matrix:")
print(f"                    Disease    No Disease")
print(f"  Test Positive      {true_positives:5d}       {false_positives:5d}     Total: {total_positive}")
print(f"  Test Negative      {false_negatives:5d}       {true_negatives:5d}")
print(f"\n  P(Disease|+) = {true_positives}/{total_positive} = {true_positives/total_positive:.4f}")
```

## Prior, Likelihood, Posterior

```python
print("\n=== PRIOR, LIKELIHOOD, POSTERIOR ===")
print("""
Bayesian thinking involves:

PRIOR: What we believed before seeing evidence
  P(hypothesis)
  
LIKELIHOOD: How probable is the evidence under each hypothesis?
  P(evidence | hypothesis)
  
POSTERIOR: Updated belief after seeing evidence
  P(hypothesis | evidence)

              Likelihood × Prior
Posterior = ─────────────────────────
             Marginal Likelihood

The posterior becomes the new prior when we see more evidence!
""")

# Example: Coin fairness
print("\nExample: Is a coin fair?")
print("You flip a coin 10 times and get 8 heads.")

# Two hypotheses
p_fair = 0.5    # Prior: 50% chance coin is fair
p_biased = 0.5  # Prior: 50% chance coin is biased (p_heads = 0.8)

from scipy.stats import binom

# Likelihood: P(8 heads in 10 | fair)
likelihood_fair = binom.pmf(8, 10, 0.5)
# Likelihood: P(8 heads in 10 | biased)
likelihood_biased = binom.pmf(8, 10, 0.8)

print(f"\nPriors: P(fair) = {p_fair}, P(biased) = {p_biased}")
print(f"Likelihoods:")
print(f"  P(8H | fair) = {likelihood_fair:.4f}")
print(f"  P(8H | biased) = {likelihood_biased:.4f}")

# Marginal likelihood
marginal = likelihood_fair * p_fair + likelihood_biased * p_biased

# Posteriors
posterior_fair = (likelihood_fair * p_fair) / marginal
posterior_biased = (likelihood_biased * p_biased) / marginal

print(f"\nPosteriors:")
print(f"  P(fair | 8H) = {posterior_fair:.4f}")
print(f"  P(biased | 8H) = {posterior_biased:.4f}")
print(f"\nEvidence favors biased coin!")
```

## Sequential Updating

```python
print("\n=== SEQUENTIAL UPDATING ===")
print("Posterior from one observation becomes prior for the next!")

# Start with equal priors
p_biased = 0.5

# Observe sequence of heads
observations = ['H', 'H', 'H', 'T', 'H', 'H', 'T', 'H', 'H', 'H']

print(f"Starting prior P(biased) = {p_biased:.4f}")
print(f"Observations: {observations}")
print("\nUpdating after each observation:")

for i, obs in enumerate(observations):
    # Likelihoods
    if obs == 'H':
        like_fair = 0.5
        like_biased = 0.8
    else:
        like_fair = 0.5
        like_biased = 0.2
    
    # Update
    numerator = like_biased * p_biased
    denominator = like_biased * p_biased + like_fair * (1 - p_biased)
    p_biased = numerator / denominator
    
    print(f"  After {obs}: P(biased) = {p_biased:.4f}")

print(f"\nFinal: {p_biased*100:.1f}% confident coin is biased")
```

## Bayes in Machine Learning

```python
print("\n=== BAYES IN MACHINE LEARNING ===")
print("""
NAIVE BAYES CLASSIFIER:
  Assumes features are independent given class
  
  P(Class | Features) ∝ P(Class) × ∏ P(Feature_i | Class)
  
SPAM FILTER EXAMPLE:
  P(Spam | "free", "money") ∝ P(Spam) × P("free"|Spam) × P("money"|Spam)
  P(Ham | "free", "money") ∝ P(Ham) × P("free"|Ham) × P("money"|Ham)
  
  Classify as whichever is higher!

BAYESIAN NEURAL NETWORKS:
  - Put distributions over weights instead of point estimates
  - Uncertainty quantification
  
BAYESIAN OPTIMIZATION:
  - Choose next hyperparameter to try
  - Balance exploration and exploitation
""")

# Simple spam filter example
print("\nSimple Spam Filter:")

# Training data statistics
p_spam = 0.3
p_ham = 0.7

# Word probabilities
p_free_spam = 0.8
p_free_ham = 0.1
p_money_spam = 0.7
p_money_ham = 0.05

# Email contains "free" and "money"
print("Email contains: 'free' and 'money'")

# Calculate posteriors
score_spam = p_spam * p_free_spam * p_money_spam
score_ham = p_ham * p_free_ham * p_money_ham

# Normalize
total = score_spam + score_ham
p_spam_posterior = score_spam / total
p_ham_posterior = score_ham / total

print(f"\nUnnormalized scores:")
print(f"  Spam: {p_spam} × {p_free_spam} × {p_money_spam} = {score_spam:.4f}")
print(f"  Ham:  {p_ham} × {p_free_ham} × {p_money_ham} = {score_ham:.4f}")
print(f"\nP(Spam | email) = {p_spam_posterior:.4f}")
print(f"P(Ham | email) = {p_ham_posterior:.4f}")
print(f"\nClassification: {'SPAM' if p_spam_posterior > p_ham_posterior else 'HAM'}")
```

## Key Points

- **Bayes' Theorem**: P(A|B) = P(B|A) × P(A) / P(B)
- **Prior**: Initial belief before evidence
- **Likelihood**: Probability of evidence given hypothesis
- **Posterior**: Updated belief after evidence
- **Base rate**: Prior probability matters! (medical testing)
- **Sequential updating**: Posterior becomes new prior
- **Applications**: Medical diagnosis, spam filtering, ML classifiers

## Reflection Questions

1. Why do doctors recommend a second test after a positive result?
2. How does the prior probability affect the posterior in Bayes' theorem?
3. What does it mean for Naive Bayes to assume feature independence?
