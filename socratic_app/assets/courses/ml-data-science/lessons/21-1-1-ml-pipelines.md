# ML Pipelines and Orchestration

## Introduction

ML pipelines automate the end-to-end machine learning workflow, from data ingestion to model deployment. Orchestration ensures these pipelines run reliably and at scale.

## Why Pipelines?

```python
import numpy as np
import pandas as pd

print("=== THE NEED FOR ML PIPELINES ===")
print("""
WITHOUT PIPELINES:
  - Manual, error-prone processes
  - Hard to reproduce results
  - Inconsistent between environments
  - Difficult to scale

WITH PIPELINES:
  ✓ Reproducibility: Same inputs → Same outputs
  ✓ Automation: Reduce manual intervention
  ✓ Scalability: Handle large datasets
  ✓ Monitoring: Track pipeline health
  ✓ Versioning: Track changes over time
  ✓ Testing: Validate each step

TYPICAL ML WORKFLOW:
  Data Ingestion → Validation → Preprocessing →
  Feature Engineering → Training → Evaluation →
  Validation → Deployment → Monitoring
""")
```

## Pipeline Components

```python
print("\n=== PIPELINE COMPONENTS ===")
print("""
1. DATA INGESTION:
   - Read from various sources
   - Handle different formats
   - Schedule regular updates

2. DATA VALIDATION:
   - Schema checking
   - Statistical validation
   - Data drift detection

3. PREPROCESSING:
   - Cleaning
   - Normalization
   - Missing value handling

4. FEATURE ENGINEERING:
   - Feature computation
   - Feature selection
   - Feature store integration

5. TRAINING:
   - Model training
   - Hyperparameter tuning
   - Experiment tracking

6. EVALUATION:
   - Performance metrics
   - Fairness evaluation
   - Comparison to baseline

7. MODEL VALIDATION:
   - Automated checks
   - A/B test analysis
   - Approval gates

8. DEPLOYMENT:
   - Model serving
   - Canary releases
   - Rollback capabilities

9. MONITORING:
   - Performance tracking
   - Drift detection
   - Alerting
""")
```

## Scikit-learn Pipelines

```python
print("\n=== SKLEARN PIPELINES ===")
print("""
Basic building block for ML pipelines in Python
Ensures consistent preprocessing between train and predict
""")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Sample data
np.random.seed(42)
n_samples = 100

X = pd.DataFrame({
    'numeric1': np.random.randn(n_samples),
    'numeric2': np.random.randn(n_samples) * 10 + 50,
    'category': np.random.choice(['A', 'B', 'C'], n_samples)
})
X.loc[np.random.choice(n_samples, 10), 'numeric1'] = np.nan

y = (X['numeric1'].fillna(0) + X['numeric2'] > 50).astype(int)

# Define preprocessing for different column types
numeric_features = ['numeric1', 'numeric2']
categorical_features = ['category']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# One-hot encoding for categorical (simplified)
from sklearn.preprocessing import OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
])

# Train and evaluate
scores = cross_val_score(full_pipeline, X, y, cv=3)
print(f"Cross-validation scores: {scores.round(3)}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Pipeline ensures same preprocessing on new data
full_pipeline.fit(X, y)
print("\nPipeline trained successfully!")
print(f"Steps: {[step[0] for step in full_pipeline.steps]}")
```

## Pipeline Orchestration Tools

```python
print("\n=== ORCHESTRATION TOOLS ===")
print("""
1. APACHE AIRFLOW:
   - DAG-based workflows
   - Rich scheduling
   - Wide integration
   - Python-native
   
   Example DAG structure:
   ingest >> validate >> [preprocess, feature_eng] >> train >> deploy

2. KUBEFLOW PIPELINES:
   - Kubernetes-native
   - ML-specific components
   - Experiment tracking
   - GPU support
   
3. PREFECT:
   - Modern Python workflows
   - Dynamic pipelines
   - Cloud-native
   
4. DAGSTER:
   - Data-aware orchestration
   - Strong typing
   - Testing-focused

5. MLflow:
   - Experiment tracking
   - Model registry
   - Deployment

6. ZenML:
   - ML-specific abstraction
   - Stack-based architecture
   - Multiple backends
""")

# Conceptual Airflow DAG
airflow_dag_example = """
# Airflow DAG Example (conceptual)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'data_science',
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

with DAG('ml_training_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:
    
    ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data_func
    )
    
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_func
    )
    
    preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess_func
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_func
    )
    
    evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate_func
    )
    
    # Define dependencies
    ingest >> validate >> preprocess >> train >> evaluate
"""

print("Airflow DAG structure example:")
print(airflow_dag_example)
```

## Data Validation

```python
print("\n=== DATA VALIDATION ===")
print("""
Validate data before training to catch issues early

SCHEMA VALIDATION:
  - Column names and types
  - Required fields
  - Value ranges

STATISTICAL VALIDATION:
  - Distribution checks
  - Outlier detection
  - Correlation stability

TOOLS:
  - Great Expectations
  - Pandera
  - TensorFlow Data Validation (TFDV)
""")

# Simple validation example
def validate_data(df, schema):
    """Basic data validation"""
    errors = []
    
    # Check required columns
    for col in schema['required_columns']:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check data types
    for col, dtype in schema.get('column_types', {}).items():
        if col in df.columns and not df[col].dtype == dtype:
            # Try to convert
            try:
                df[col].astype(dtype)
            except:
                errors.append(f"Column {col} should be {dtype}")
    
    # Check ranges
    for col, (min_val, max_val) in schema.get('ranges', {}).items():
        if col in df.columns:
            if df[col].min() < min_val or df[col].max() > max_val:
                errors.append(f"Column {col} out of range [{min_val}, {max_val}]")
    
    # Check null thresholds
    for col, threshold in schema.get('null_thresholds', {}).items():
        if col in df.columns:
            null_rate = df[col].isnull().mean()
            if null_rate > threshold:
                errors.append(f"Column {col} null rate {null_rate:.2%} exceeds {threshold:.2%}")
    
    return len(errors) == 0, errors

# Example schema
schema = {
    'required_columns': ['numeric1', 'numeric2', 'category'],
    'column_types': {'category': 'object'},
    'ranges': {'numeric2': (0, 100)},
    'null_thresholds': {'numeric1': 0.2}
}

valid, errors = validate_data(X, schema)
print(f"Data valid: {valid}")
if errors:
    for e in errors:
        print(f"  - {e}")
```

## Pipeline Best Practices

```python
print("\n=== BEST PRACTICES ===")
print("""
1. IDEMPOTENCY:
   Running pipeline twice gives same result
   - Use deterministic seeds
   - Avoid side effects
   - Clean up before writing

2. MODULARITY:
   Small, focused components
   - Single responsibility
   - Reusable across pipelines
   - Easy to test

3. VERSIONING:
   Track everything
   - Code version (git)
   - Data version (DVC)
   - Model version
   - Config version

4. TESTING:
   Test each component
   - Unit tests for functions
   - Integration tests for pipeline
   - Data quality tests

5. MONITORING:
   Know when things break
   - Pipeline success/failure
   - Data quality metrics
   - Model performance

6. DOCUMENTATION:
   Explain the pipeline
   - Input/output schemas
   - Dependencies
   - Assumptions
""")

# Testing example
def test_preprocessing_step():
    """Unit test for preprocessing"""
    # Input
    test_data = pd.DataFrame({
        'numeric1': [1.0, 2.0, np.nan],
        'numeric2': [10.0, 20.0, 30.0]
    })
    
    # Expected output after imputation with median
    expected_values = [1.0, 2.0, 1.5]  # median of [1, 2] = 1.5
    
    # Run preprocessing
    imputer = SimpleImputer(strategy='median')
    result = imputer.fit_transform(test_data[['numeric1']])
    
    # Assert
    np.testing.assert_array_almost_equal(result.flatten(), expected_values)
    print("✓ Preprocessing test passed")

test_preprocessing_step()
```

## Key Points

- **Pipelines**: Automate end-to-end ML workflow
- **Components**: Ingestion, validation, preprocessing, training, deployment
- **Orchestration**: Airflow, Kubeflow, Prefect for scheduling and dependencies
- **Validation**: Check data schema and statistics before training
- **Best practices**: Idempotency, modularity, versioning, testing
- **sklearn Pipeline**: Basic building block for preprocessing + model

## Reflection Questions

1. What are the risks of not using a formal ML pipeline?
2. How would you decide between Airflow and Kubeflow for your project?
3. What data validation checks are most important for your models?
