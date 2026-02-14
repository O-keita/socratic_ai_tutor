# Algorithmic Bias and Fairness

## Introduction

Machine learning systems can perpetuate or amplify societal biases. Understanding and mitigating bias is crucial for building fair and trustworthy AI systems.

## Sources of Bias

```python
import numpy as np
import pandas as pd

print("=== SOURCES OF BIAS IN ML ===")
print("""
1. HISTORICAL BIAS:
   Data reflects past inequalities
   Example: Hiring data favors demographics historically hired

2. REPRESENTATION BIAS:
   Training data doesn't represent all groups equally
   Example: Face recognition less accurate for minorities

3. MEASUREMENT BIAS:
   Features are proxies that differ across groups
   Example: Credit scores affected by systemic factors

4. AGGREGATION BIAS:
   One model for all, when groups differ
   Example: Medical model trained mostly on one demographic

5. EVALUATION BIAS:
   Benchmarks don't represent deployment context
   Example: Testing only on English speakers

6. DEPLOYMENT BIAS:
   Model used differently than intended
   Example: Screening tool becomes decision-maker
""")
```

## Fairness Definitions

```python
print("\n=== FAIRNESS METRICS ===")
print("""
Protected attribute: Sensitive characteristic (race, gender, age)
Groups: Subpopulations based on protected attribute

KEY DEFINITIONS:

1. DEMOGRAPHIC PARITY (Statistical Parity):
   P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
   "Equal positive rates across groups"
   
   Problem: Ignores actual qualifications

2. EQUALIZED ODDS:
   P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)  (True positive rate)
   P(Ŷ=1|Y=0,A=0) = P(Ŷ=1|Y=0,A=1)  (False positive rate)
   "Equal error rates across groups"

3. EQUAL OPPORTUNITY:
   P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
   "Equal true positive rates"
   Qualified individuals have equal chance

4. PREDICTIVE PARITY:
   P(Y=1|Ŷ=1,A=0) = P(Y=1|Ŷ=1,A=1)
   "Equal precision across groups"

5. INDIVIDUAL FAIRNESS:
   Similar individuals get similar predictions
   d(f(x₁), f(x₂)) ≤ d(x₁, x₂)
""")

def compute_fairness_metrics(y_true, y_pred, protected):
    """Compute common fairness metrics"""
    groups = np.unique(protected)
    
    metrics = {}
    
    for g in groups:
        mask = protected == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        # Positive rate
        positive_rate = np.mean(y_p == 1)
        
        # True positive rate (if positive exists)
        if np.sum(y_t == 1) > 0:
            tpr = np.sum((y_p == 1) & (y_t == 1)) / np.sum(y_t == 1)
        else:
            tpr = 0
        
        # False positive rate (if negative exists)
        if np.sum(y_t == 0) > 0:
            fpr = np.sum((y_p == 1) & (y_t == 0)) / np.sum(y_t == 0)
        else:
            fpr = 0
        
        metrics[g] = {
            'positive_rate': positive_rate,
            'tpr': tpr,
            'fpr': fpr
        }
    
    return metrics

# Example
np.random.seed(42)
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
protected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

metrics = compute_fairness_metrics(y_true, y_pred, protected)

print("Fairness metrics by group:")
for group, m in metrics.items():
    print(f"\nGroup {group}:")
    print(f"  Positive rate: {m['positive_rate']:.2f}")
    print(f"  True positive rate: {m['tpr']:.2f}")
    print(f"  False positive rate: {m['fpr']:.2f}")

# Disparities
print("\nDisparities:")
print(f"  Demographic parity gap: {abs(metrics[0]['positive_rate'] - metrics[1]['positive_rate']):.2f}")
print(f"  Equal opportunity gap: {abs(metrics[0]['tpr'] - metrics[1]['tpr']):.2f}")
```

## Impossibility Results

```python
print("\n=== FAIRNESS IMPOSSIBILITIES ===")
print("""
IMPORTANT: Cannot satisfy all fairness criteria simultaneously!

Chouldechova's Impossibility Theorem:
  If base rates differ between groups:
  P(Y=1|A=0) ≠ P(Y=1|A=1)
  
  Then CANNOT have both:
  - Predictive parity
  - Equal false positive AND false negative rates

Kleinberg's Impossibility:
  Cannot have all three:
  - Calibration
  - Balance for positive class
  - Balance for negative class
  (unless perfect prediction or equal base rates)

IMPLICATIONS:
  - Must CHOOSE which fairness criterion matters most
  - Context-dependent decisions
  - No "one-size-fits-all" solution
  - Trade-offs are fundamental, not technical failures
""")
```

## Bias Mitigation Strategies

```python
print("\n=== MITIGATION APPROACHES ===")
print("""
1. PRE-PROCESSING (Data):
   - Resampling: Balance representation
   - Reweighting: Adjust sample weights
   - Transformation: Remove bias from features

2. IN-PROCESSING (Model):
   - Fairness constraints in optimization
   - Adversarial debiasing
   - Regularization toward fairness

3. POST-PROCESSING (Predictions):
   - Threshold adjustment per group
   - Calibration per group
   - Reject option classification
""")

# Example: Threshold adjustment
def equalize_opportunity(y_scores, y_true, protected, target_tpr=0.8):
    """Adjust thresholds to equalize TPR"""
    groups = np.unique(protected)
    thresholds = {}
    
    for g in groups:
        mask = (protected == g) & (y_true == 1)
        scores_g = y_scores[mask]
        
        if len(scores_g) == 0:
            thresholds[g] = 0.5
            continue
        
        # Find threshold for target TPR
        sorted_scores = np.sort(scores_g)[::-1]
        idx = int(len(sorted_scores) * target_tpr)
        thresholds[g] = sorted_scores[min(idx, len(sorted_scores)-1)]
    
    return thresholds

# Simulated scores
np.random.seed(42)
y_scores = np.random.rand(100)
y_true = (y_scores > 0.5).astype(int) + np.random.randint(-1, 2, 100).clip(0, 1)
y_true = np.clip(y_true, 0, 1)
protected = np.random.randint(0, 2, 100)

thresholds = equalize_opportunity(y_scores, y_true, protected)
print("Group-specific thresholds for equal opportunity:")
for g, t in thresholds.items():
    print(f"  Group {g}: {t:.3f}")
```

## Adversarial Debiasing

```python
print("\n=== ADVERSARIAL DEBIASING ===")
print("""
Use adversarial training to remove protected attribute info

Architecture:
  Input → Encoder → Representation → Predictor → Ŷ
                          ↓
                    Adversary → Â (predicts protected)

Training:
  - Predictor: Minimize prediction loss
  - Adversary: Predict protected attribute from representation
  - Encoder: Maximize adversary loss (confuse adversary)

Result: Representation doesn't encode protected attribute

Loss = L_pred - λ × L_adversary

Higher λ → More fairness, potentially less accuracy
""")

print("""
Pseudo-code:

class AdversarialDebiasing:
    def __init__(self):
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.adversary = Adversary()
    
    def train_step(self, X, y, protected):
        # Forward pass
        representation = self.encoder(X)
        y_pred = self.predictor(representation)
        a_pred = self.adversary(representation.detach())
        
        # Predictor loss
        pred_loss = CrossEntropy(y_pred, y)
        
        # Adversary loss
        adv_loss = CrossEntropy(a_pred, protected)
        
        # Adversarial loss (encoder wants adversary to fail)
        adv_loss_for_encoder = CrossEntropy(
            self.adversary(representation), 
            protected
        )
        
        # Update predictor and encoder
        encoder_loss = pred_loss - lambda * adv_loss_for_encoder
        
        # Update adversary separately
        update(self.adversary, adv_loss)
        update(self.encoder, self.predictor, encoder_loss)
""")
```

## Fairness in Practice

```python
print("\n=== PRACTICAL CONSIDERATIONS ===")
print("""
1. PROTECTED ATTRIBUTES:
   - May not be available (privacy)
   - Proxy features exist (zip code → race)
   - Multi-dimensional (intersectionality)

2. CHOOSING FAIRNESS METRIC:
   - Context matters: Hiring vs. lending vs. criminal justice
   - Stakeholder input essential
   - Consider harm of different errors

3. AUDITING:
   - Regular fairness audits
   - Subgroup analysis
   - Monitoring in production

4. DOCUMENTATION:
   - Model cards
   - Datasheets for datasets
   - Impact assessments

5. LEGAL REQUIREMENTS:
   - Varies by jurisdiction
   - Disparate impact (US)
   - GDPR (EU)
""")

print("""
FAIRNESS CHECKLIST:

Before Training:
□ Assess potential harms
□ Define fairness requirements
□ Audit data for representation
□ Identify protected attributes

During Training:
□ Choose appropriate fairness metric
□ Implement fairness constraints
□ Monitor subgroup performance

After Training:
□ Fairness audit across subgroups
□ Document limitations
□ Plan for monitoring
□ Establish feedback channels
""")
```

## Key Points

- **Bias sources**: Historical, representation, measurement, aggregation
- **Fairness metrics**: Demographic parity, equalized odds, equal opportunity
- **Impossibility**: Can't satisfy all fairness criteria simultaneously
- **Mitigation**: Pre-processing, in-processing, post-processing
- **Context matters**: Choose fairness metric based on application
- **Ongoing process**: Regular auditing and monitoring required

## Reflection Questions

1. Why is it impossible to achieve all fairness criteria simultaneously?
2. When might demographic parity be the wrong fairness metric?
3. How should organizations choose between conflicting fairness goals?
