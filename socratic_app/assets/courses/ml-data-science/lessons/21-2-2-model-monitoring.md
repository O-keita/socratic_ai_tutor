# Model Monitoring and Maintenance

## Introduction

Deployed models require continuous monitoring to ensure they perform well over time. Model performance can degrade due to data drift, concept drift, or system issues.

## Why Monitor?

```python
import numpy as np
import pandas as pd

print("=== WHY MODEL MONITORING? ===")
print("""
MODELS DEGRADE OVER TIME:

1. DATA DRIFT:
   Input distribution changes
   - User behavior shifts
   - New products/categories
   - Seasonal variations

2. CONCEPT DRIFT:
   Relationship between X and y changes
   - Market dynamics shift
   - User preferences evolve
   - World events (pandemic, regulations)

3. SYSTEM ISSUES:
   Technical problems
   - Feature pipeline failures
   - Data quality issues
   - Infrastructure problems

WITHOUT MONITORING:
  "Silent failures" - model gives wrong predictions
  No visibility into degradation
  Problems discovered too late

WITH MONITORING:
  ✓ Early detection of issues
  ✓ Automated alerts
  ✓ Data for debugging
  ✓ Compliance and audit trails
""")
```

## What to Monitor

```python
print("\n=== MONITORING METRICS ===")
print("""
1. MODEL PERFORMANCE:
   - Accuracy, precision, recall, F1
   - AUC-ROC, AUC-PR
   - RMSE, MAE (regression)
   - Custom business metrics

2. DATA QUALITY:
   - Missing values
   - Schema violations
   - Outlier frequency
   - Feature statistics

3. INPUT DATA DISTRIBUTION:
   - Feature means, std, quantiles
   - Categorical distributions
   - Correlation stability
   - New categories

4. PREDICTION DISTRIBUTION:
   - Prediction mean and spread
   - Class balance
   - Confidence distribution
   - Edge cases

5. OPERATIONAL METRICS:
   - Latency (p50, p95, p99)
   - Throughput (requests/sec)
   - Error rates
   - Resource usage

6. BUSINESS METRICS:
   - Conversion rate
   - Revenue impact
   - User engagement
   - Customer complaints
""")
```

## Detecting Data Drift

```python
print("\n=== DATA DRIFT DETECTION ===")
print("""
Statistical tests to detect distribution changes

NUMERICAL FEATURES:
  - Kolmogorov-Smirnov test
  - Population Stability Index (PSI)
  - Jensen-Shannon divergence
  - Mean/variance monitoring

CATEGORICAL FEATURES:
  - Chi-squared test
  - PSI for categories
  - New category detection

MULTIVARIATE:
  - Maximum Mean Discrepancy (MMD)
  - Domain classifier approach
""")

def population_stability_index(expected, actual, bins=10):
    """Calculate PSI for a feature"""
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Count in each bin
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]
    
    # Convert to proportions
    expected_prop = expected_counts / len(expected)
    actual_prop = actual_counts / len(actual)
    
    # Avoid division by zero
    expected_prop = np.clip(expected_prop, 0.001, None)
    actual_prop = np.clip(actual_prop, 0.001, None)
    
    # PSI calculation
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    
    return psi

# Example: Detect drift
np.random.seed(42)
training_data = np.random.normal(0, 1, 1000)
production_data_no_drift = np.random.normal(0, 1, 1000)
production_data_with_drift = np.random.normal(0.5, 1.2, 1000)

psi_no_drift = population_stability_index(training_data, production_data_no_drift)
psi_with_drift = population_stability_index(training_data, production_data_with_drift)

print("PSI values:")
print(f"  No drift: {psi_no_drift:.4f}")
print(f"  With drift: {psi_with_drift:.4f}")
print("\nInterpretation:")
print("  PSI < 0.1: No significant change")
print("  0.1 ≤ PSI < 0.2: Moderate change")
print("  PSI ≥ 0.2: Significant change - investigate!")

def ks_test(expected, actual):
    """Kolmogorov-Smirnov test for drift"""
    from scipy import stats
    statistic, p_value = stats.ks_2samp(expected, actual)
    return statistic, p_value

# KS test
ks_stat, p_value = ks_test(training_data, production_data_with_drift)
print(f"\nKS test p-value: {p_value:.6f}")
print("  p < 0.05 suggests significant drift")
```

## Concept Drift Detection

```python
print("\n=== CONCEPT DRIFT DETECTION ===")
print("""
Concept drift: P(Y|X) changes

DETECTION APPROACHES:

1. PERFORMANCE MONITORING:
   Track accuracy over time
   - Rolling window metrics
   - Sudden drops indicate drift
   
2. LABEL DELAY HANDLING:
   When ground truth is delayed:
   - Monitor prediction confidence
   - Track proxy metrics
   - Set up delayed evaluation

3. DRIFT DETECTION ALGORITHMS:
   - ADWIN (Adaptive Windowing)
   - DDM (Drift Detection Method)
   - EDDM (Early Drift Detection Method)
   - Page-Hinkley test
""")

class DriftDetector:
    """Simple drift detection using performance degradation"""
    
    def __init__(self, window_size=100, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_accuracy = None
        self.recent_accuracies = []
    
    def set_baseline(self, accuracy):
        self.baseline_accuracy = accuracy
        print(f"Baseline accuracy set to {accuracy:.4f}")
    
    def update(self, accuracy):
        self.recent_accuracies.append(accuracy)
        if len(self.recent_accuracies) > self.window_size:
            self.recent_accuracies.pop(0)
        
        return self.check_drift()
    
    def check_drift(self):
        if len(self.recent_accuracies) < self.window_size // 2:
            return False, 0
        
        recent_mean = np.mean(self.recent_accuracies)
        degradation = self.baseline_accuracy - recent_mean
        
        drift_detected = degradation > self.threshold
        
        return drift_detected, degradation

# Demo
detector = DriftDetector(window_size=10, threshold=0.03)
detector.set_baseline(0.85)

# Simulate accuracy measurements
accuracies = [0.84, 0.85, 0.83, 0.86, 0.84, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76]

print("\nMonitoring over time:")
for i, acc in enumerate(accuracies):
    drift, degradation = detector.update(acc)
    status = "⚠️ DRIFT" if drift else "OK"
    print(f"  Step {i}: accuracy={acc:.2f}, degradation={degradation:.3f}, status={status}")
```

## Alerting and Response

```python
print("\n=== ALERTING SYSTEM ===")
print("""
ALERT LEVELS:

INFO: Noteworthy but not urgent
  - Minor metric fluctuation
  - New data patterns observed

WARNING: Investigate soon
  - Moderate drift detected
  - Performance declining
  - Increased error rate

CRITICAL: Immediate action needed
  - Severe performance drop
  - System errors
  - Data pipeline failure

RESPONSE PLAYBOOK:

1. TRIAGE
   - Assess impact severity
   - Check for system issues
   - Review recent changes

2. INVESTIGATE
   - Analyze drift metrics
   - Check data quality
   - Review prediction samples

3. MITIGATE
   - Rollback if needed
   - Adjust thresholds
   - Enable fallback

4. FIX
   - Retrain model
   - Update features
   - Fix data pipeline

5. DOCUMENT
   - Root cause analysis
   - Prevention measures
   - Update monitoring
""")

class AlertSystem:
    def __init__(self):
        self.thresholds = {
            'accuracy_drop': {'warning': 0.03, 'critical': 0.1},
            'latency_ms': {'warning': 200, 'critical': 500},
            'error_rate': {'warning': 0.01, 'critical': 0.05}
        }
    
    def check_metric(self, metric_name, value, baseline=None):
        if metric_name == 'accuracy_drop':
            value = baseline - value if baseline else 0
        
        thresholds = self.thresholds.get(metric_name, {})
        
        if value >= thresholds.get('critical', float('inf')):
            return 'CRITICAL', f"{metric_name}={value}"
        elif value >= thresholds.get('warning', float('inf')):
            return 'WARNING', f"{metric_name}={value}"
        else:
            return 'OK', f"{metric_name}={value}"
    
    def generate_alert(self, level, message):
        if level == 'CRITICAL':
            # Would trigger: PagerDuty, Slack, email
            print(f"🚨 CRITICAL: {message}")
        elif level == 'WARNING':
            # Would trigger: Slack, email
            print(f"⚠️ WARNING: {message}")
        else:
            print(f"✅ OK: {message}")

# Demo
alerts = AlertSystem()

print("\nAlert system demo:")
level, msg = alerts.check_metric('latency_ms', 150)
alerts.generate_alert(level, msg)

level, msg = alerts.check_metric('latency_ms', 300)
alerts.generate_alert(level, msg)

level, msg = alerts.check_metric('error_rate', 0.08)
alerts.generate_alert(level, msg)
```

## Monitoring Dashboard

```python
print("\n=== MONITORING DASHBOARD ===")
print("""
KEY COMPONENTS:

1. REAL-TIME METRICS:
   ┌────────────────────────────────────────┐
   │ Requests/sec: 1,234    Errors: 0.1%    │
   │ P50 Latency: 45ms     P99 Latency: 180ms│
   └────────────────────────────────────────┘

2. PERFORMANCE TRENDS:
   Accuracy over last 24h
   ▁▁▂▂▂▃▃▃▃▃▄▄▄▄▃▃▃▂▂▁▁▁▁▁
   
3. DRIFT INDICATORS:
   Feature    PSI     Status
   age        0.02    ✅
   income     0.08    ✅
   location   0.15    ⚠️
   
4. ALERT HISTORY:
   12:00 - WARNING: income PSI > 0.1
   09:30 - INFO: New category detected
   
5. MODEL VERSIONS:
   v1.2 (production): 95% traffic
   v1.3 (canary): 5% traffic

TOOLS:
  - Grafana + Prometheus
  - Datadog
  - MLflow
  - Evidently AI
  - WhyLabs
  - Fiddler
""")
```

## Model Retraining

```python
print("\n=== RETRAINING STRATEGIES ===")
print("""
WHEN TO RETRAIN:

1. SCHEDULED:
   - Fixed intervals (daily, weekly)
   - Simple but may be wasteful

2. TRIGGERED:
   - When drift exceeds threshold
   - Performance drops
   - New data volume reaches threshold

3. CONTINUOUS:
   - Online learning
   - Update with each new sample
   - Complex but most adaptive

RETRAINING PIPELINE:

Trigger → Collect Data → Train → Evaluate → Validate → Deploy

                                     ↓ If fails
                              Keep current model

VALIDATION BEFORE DEPLOYMENT:
  - Performance on holdout set
  - Performance on recent data
  - Fairness checks
  - Latency requirements
  - Business metric simulation
""")

class RetrainingScheduler:
    def __init__(self, drift_threshold=0.1, min_samples=1000):
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.last_retrain = None
        self.new_samples = 0
    
    def should_retrain(self, drift_score, performance_drop, new_samples):
        """Decide if retraining is needed"""
        self.new_samples += new_samples
        
        reasons = []
        
        # Check drift
        if drift_score > self.drift_threshold:
            reasons.append(f"Drift: {drift_score:.3f}")
        
        # Check performance
        if performance_drop > 0.05:
            reasons.append(f"Performance drop: {performance_drop:.3f}")
        
        # Check data volume
        if self.new_samples >= self.min_samples:
            reasons.append(f"New samples: {self.new_samples}")
        
        if reasons:
            print(f"Retraining triggered: {', '.join(reasons)}")
            return True
        
        return False
    
    def reset_after_retrain(self):
        self.new_samples = 0
        self.last_retrain = pd.Timestamp.now()

scheduler = RetrainingScheduler()
print("\nRetraining decisions:")
print(f"  Check 1: {scheduler.should_retrain(0.05, 0.02, 500)}")  # No
print(f"  Check 2: {scheduler.should_retrain(0.15, 0.02, 300)}")  # Yes (drift)
```

## Key Points

- **Monitor continuously**: Models degrade over time
- **Track multiple metrics**: Performance, data quality, operational
- **Detect drift early**: PSI, KS test, performance monitoring
- **Set up alerts**: Warning and critical thresholds
- **Plan retraining**: Scheduled, triggered, or continuous
- **Validate before deploy**: Don't make things worse

## Reflection Questions

1. How do you monitor model performance when labels are delayed?
2. What's the cost of over-retraining vs. under-retraining?
3. How do you distinguish between data drift and system bugs?
