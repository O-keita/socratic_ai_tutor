# Experiment Tracking and Model Registry

## Introduction

Experiment tracking captures the details of ML experiments for reproducibility and comparison. Model registries manage the lifecycle of trained models from development to production.

## Why Track Experiments?

```python
import numpy as np
import pandas as pd

print("=== THE EXPERIMENT TRACKING PROBLEM ===")
print("""
WITHOUT TRACKING:
  "Which model was best again?"
  "What hyperparameters did I use?"
  "Which dataset version was this trained on?"
  "Can I reproduce this result?"

You end up with:
  - model_v1.pkl
  - model_v2_final.pkl
  - model_v2_final_ACTUAL.pkl
  - model_v2_final_ACTUAL_fixed.pkl
  ...

WITH EXPERIMENT TRACKING:
  ✓ Every run logged automatically
  ✓ Parameters, metrics, artifacts saved
  ✓ Easy comparison across experiments
  ✓ Full reproducibility
  ✓ Team collaboration
""")
```

## What to Track

```python
print("\n=== WHAT TO TRACK ===")
print("""
1. CODE VERSION:
   - Git commit hash
   - Branch name
   - Diff from main

2. DATA VERSION:
   - Dataset name/path
   - Data hash/version
   - Preprocessing applied

3. PARAMETERS:
   - Hyperparameters
   - Random seeds
   - Configuration files

4. ENVIRONMENT:
   - Python version
   - Package versions
   - Hardware (GPU, CPU)

5. METRICS:
   - Training metrics over time
   - Validation metrics
   - Test metrics
   - Custom metrics

6. ARTIFACTS:
   - Trained model files
   - Plots and figures
   - Evaluation reports
   - Feature importance

7. METADATA:
   - Experiment name/tags
   - Run duration
   - Timestamp
   - Author
""")
```

## MLflow Basics

```python
print("\n=== MLFLOW EXPERIMENT TRACKING ===")
print("""
MLflow: Popular open-source platform

Components:
  - Tracking: Log parameters, metrics, artifacts
  - Projects: Reproducible runs
  - Models: Model packaging
  - Registry: Model versioning

Basic Usage:
""")

# MLflow example (conceptual - would need mlflow installed)
mlflow_example = """
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("loan_default_prediction")

with mlflow.start_run(run_name="random_forest_v1"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 5)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_auc", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")

# View experiments: mlflow ui
"""

print(mlflow_example)

# Simulated experiment tracking
class SimpleExperimentTracker:
    def __init__(self):
        self.experiments = {}
        self.current_run = None
    
    def start_run(self, run_name):
        self.current_run = {
            'name': run_name,
            'params': {},
            'metrics': {},
            'artifacts': [],
            'timestamp': pd.Timestamp.now()
        }
        return self
    
    def log_param(self, key, value):
        self.current_run['params'][key] = value
    
    def log_metric(self, key, value):
        self.current_run['metrics'][key] = value
    
    def log_artifact(self, path):
        self.current_run['artifacts'].append(path)
    
    def end_run(self):
        run_id = len(self.experiments)
        self.experiments[run_id] = self.current_run
        self.current_run = None
        return run_id
    
    def compare_runs(self, run_ids):
        comparison = []
        for run_id in run_ids:
            run = self.experiments[run_id]
            comparison.append({
                'run_id': run_id,
                'name': run['name'],
                **run['params'],
                **run['metrics']
            })
        return pd.DataFrame(comparison)

# Demo
tracker = SimpleExperimentTracker()

# Run 1
tracker.start_run("rf_baseline")
tracker.log_param("n_estimators", 100)
tracker.log_param("max_depth", 10)
tracker.log_metric("accuracy", 0.85)
tracker.log_metric("auc", 0.89)
run1 = tracker.end_run()

# Run 2
tracker.start_run("rf_tuned")
tracker.log_param("n_estimators", 200)
tracker.log_param("max_depth", 15)
tracker.log_metric("accuracy", 0.87)
tracker.log_metric("auc", 0.91)
run2 = tracker.end_run()

print("\nExperiment comparison:")
print(tracker.compare_runs([run1, run2]).to_string(index=False))
```

## Model Registry

```python
print("\n=== MODEL REGISTRY ===")
print("""
Centralized repository for managing ML models

KEY FEATURES:
  - Model versioning
  - Stage transitions (dev → staging → production)
  - Approval workflows
  - Model lineage
  - Deployment integration

MODEL STAGES:
  1. None: Just logged
  2. Staging: Under testing
  3. Production: Serving live traffic
  4. Archived: No longer in use

WORKFLOW:
  1. Train and log model
  2. Register in model registry
  3. Transition to staging
  4. Run tests (A/B, shadow mode)
  5. Approve for production
  6. Deploy
  7. Monitor
  8. Archive when replaced
""")

# MLflow Model Registry example
mlflow_registry_example = """
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model (from run)
result = mlflow.register_model(
    "runs:/abc123/model",
    "loan_default_model"
)

# Transition to staging
client.transition_model_version_stage(
    name="loan_default_model",
    version=1,
    stage="Staging"
)

# Add description
client.update_model_version(
    name="loan_default_model",
    version=1,
    description="Random Forest with improved feature engineering"
)

# Get production model
prod_model = mlflow.pyfunc.load_model(
    "models:/loan_default_model/Production"
)

# Promote to production
client.transition_model_version_stage(
    name="loan_default_model",
    version=1,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="loan_default_model",
    version=0,
    stage="Archived"
)
"""

print(mlflow_registry_example)
```

## Experiment Comparison

```python
print("\n=== COMPARING EXPERIMENTS ===")
print("""
COMPARISON TECHNIQUES:

1. Metric Tables:
   Compare key metrics across runs
   
2. Learning Curves:
   Training/validation loss over epochs
   
3. Parallel Coordinates:
   Visualize hyperparameter relationships
   
4. Scatter Plots:
   Parameter vs metric relationships

5. Statistical Tests:
   Significance of improvements
""")

# Simulated experiment results
np.random.seed(42)
experiments = pd.DataFrame({
    'model': ['RF', 'RF', 'RF', 'XGB', 'XGB', 'XGB', 'NN', 'NN', 'NN'],
    'learning_rate': [None, None, None, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.01],
    'max_depth': [5, 10, 15, 5, 10, 15, None, None, None],
    'accuracy': [0.82, 0.85, 0.84, 0.86, 0.88, 0.87, 0.83, 0.81, 0.89],
    'auc': [0.85, 0.88, 0.87, 0.89, 0.92, 0.91, 0.86, 0.84, 0.93],
    'train_time_sec': [10, 25, 45, 15, 30, 60, 120, 120, 300]
})

print("Experiment results:")
print(experiments.to_string(index=False))

# Best by metric
print("\nBest experiments:")
best_accuracy = experiments.loc[experiments['accuracy'].idxmax()]
print(f"Best accuracy: {best_accuracy['model']} with {best_accuracy['accuracy']:.2f}")

best_auc = experiments.loc[experiments['auc'].idxmax()]
print(f"Best AUC: {best_auc['model']} with {best_auc['auc']:.2f}")

# Pareto optimal (accuracy vs train_time)
print("\nPareto analysis (accuracy vs train_time):")
pareto_optimal = []
for i, row in experiments.iterrows():
    dominated = False
    for j, other in experiments.iterrows():
        if (other['accuracy'] > row['accuracy'] and 
            other['train_time_sec'] <= row['train_time_sec']):
            dominated = True
            break
    if not dominated:
        pareto_optimal.append(row['model'])
print(f"Pareto optimal models: {set(pareto_optimal)}")
```

## Reproducibility

```python
print("\n=== ENSURING REPRODUCIBILITY ===")
print("""
CHECKLIST FOR REPRODUCIBLE EXPERIMENTS:

□ Version Control
  - Code in git
  - Commit hash logged
  - No uncommitted changes

□ Data Versioning
  - Data hash or version ID
  - DVC for large files
  - Track preprocessing

□ Environment
  - requirements.txt or conda env
  - Docker container
  - Hardware info logged

□ Random Seeds
  - Set all seeds explicitly
  - numpy, random, torch, sklearn

□ Configuration
  - Config files checked in
  - No hardcoded values
  - Environment variables documented

□ Documentation
  - README with setup instructions
  - Example commands
  - Known issues
""")

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    # If using PyTorch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    
    # If using TensorFlow
    # tf.random.set_seed(seed)
    
    print(f"All seeds set to {seed}")

set_all_seeds(42)

# Environment capture
def capture_environment():
    """Capture environment information"""
    import sys
    import platform
    
    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
    }
    
    # Would also capture:
    # - GPU info (nvidia-smi)
    # - Package versions (pip freeze)
    # - Git info
    
    return env_info

env = capture_environment()
print("\nEnvironment captured:")
for key, value in env.items():
    print(f"  {key}: {value[:50]}...")
```

## Key Points

- **Track everything**: Parameters, metrics, artifacts, code, data, environment
- **MLflow**: Popular platform for tracking and registry
- **Model stages**: Development → Staging → Production → Archived
- **Comparison**: Tables, visualizations, statistical tests
- **Reproducibility**: Seeds, versions, documentation
- **Collaboration**: Shared experiments, consistent tracking

## Reflection Questions

1. What happens when you can't reproduce a model that's in production?
2. How do you decide when a new model is good enough to promote to production?
3. What's the minimum viable experiment tracking for a small project?
