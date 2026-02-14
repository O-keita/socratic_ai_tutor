# AutoML and Neural Architecture Search

## Introduction

Automated Machine Learning (AutoML) automates the process of selecting algorithms, tuning hyperparameters, and even designing neural network architectures, making ML more accessible and efficient.

## What is AutoML?

```python
import numpy as np
import pandas as pd

print("=== AUTOML OVERVIEW ===")
print("""
AutoML automates the ML pipeline:

TRADITIONAL ML WORKFLOW:
  Data → Feature Eng → Model Select → Hyperparameter Tune → Evaluate
         (manual)       (manual)         (manual)

AUTOML WORKFLOW:
  Data → AutoML System → Best Model
         (automated)

WHAT AUTOML AUTOMATES:
  1. Feature preprocessing
  2. Feature engineering
  3. Algorithm selection
  4. Hyperparameter tuning
  5. Model ensembling
  6. Neural architecture design

BENEFITS:
  ✓ Faster development
  ✓ Better performance (often)
  ✓ Less expertise required
  ✓ Reproducible process

LIMITATIONS:
  ✗ Black box process
  ✗ Compute intensive
  ✗ May overfit validation
  ✗ Limited customization
""")
```

## Hyperparameter Optimization

```python
print("\n=== HYPERPARAMETER OPTIMIZATION ===")
print("""
Finding best hyperparameters automatically

METHODS:

1. GRID SEARCH:
   Try all combinations
   - Exhaustive but expensive
   - O(n^k) for k parameters
   
2. RANDOM SEARCH:
   Sample randomly from distributions
   - More efficient than grid
   - Better for high dimensions
   
3. BAYESIAN OPTIMIZATION:
   Build surrogate model of objective
   - Smart sampling based on past results
   - More sample efficient
   - Uses acquisition function (EI, UCB)
   
4. EVOLUTIONARY:
   Genetic algorithms for search
   - Population of configurations
   - Mutation and crossover
   
5. HYPERBAND / ASHA:
   Early stopping of bad configs
   - Train many configs briefly
   - Keep promising ones
   - Very efficient
""")

# Simple Bayesian-like optimization demo
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def random_search(X, y, param_distributions, n_iter=20):
    """Simple random search"""
    results = []
    
    for i in range(n_iter):
        # Sample parameters
        params = {}
        for param, dist in param_distributions.items():
            if isinstance(dist, list):
                params[param] = np.random.choice(dist)
            else:
                params[param] = np.random.uniform(*dist)
        
        # Evaluate
        model = RandomForestClassifier(
            n_estimators=int(params.get('n_estimators', 100)),
            max_depth=int(params.get('max_depth', 10)),
            random_state=42
        )
        
        scores = cross_val_score(model, X, y, cv=3)
        mean_score = scores.mean()
        
        results.append({**params, 'score': mean_score})
        
        if (i + 1) % 5 == 0:
            print(f"Iteration {i+1}: best score = {max(r['score'] for r in results):.4f}")
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

# Demo
np.random.seed(42)
X = np.random.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 15, 20]
}

print("Running random search...")
results = random_search(X, y, param_distributions, n_iter=10)

print("\nTop 3 configurations:")
for i, r in enumerate(results[:3]):
    print(f"  {i+1}. n_estimators={r['n_estimators']}, max_depth={r['max_depth']}, score={r['score']:.4f}")
```

## AutoML Frameworks

```python
print("\n=== AUTOML FRAMEWORKS ===")
print("""
1. AUTO-SKLEARN:
   - Scikit-learn wrapper
   - Meta-learning for warm start
   - Ensemble of best models
   
   from autosklearn.classification import AutoSklearnClassifier
   automl = AutoSklearnClassifier(time_left_for_this_task=3600)
   automl.fit(X_train, y_train)

2. TPOT:
   - Genetic algorithm
   - Exports sklearn code
   - Good interpretability
   
   from tpot import TPOTClassifier
   tpot = TPOTClassifier(generations=5, population_size=50)
   tpot.fit(X_train, y_train)
   tpot.export('best_pipeline.py')

3. H2O AUTOML:
   - Distributed computing
   - Good for large datasets
   - Automatic ensembling
   
   import h2o
   from h2o.automl import H2OAutoML
   aml = H2OAutoML(max_runtime_secs=3600)
   aml.train(x=features, y=target, training_frame=train)

4. GOOGLE CLOUD AUTOML:
   - Managed service
   - Neural architecture search
   - Pre-trained models

5. AZURE AUTOML:
   - Integrated with Azure ML
   - Explainability built-in
   - Enterprise features

6. AMAZON SAGEMAKER AUTOPILOT:
   - AWS integration
   - Transparency into process
   - Custom algorithms
""")
```

## Neural Architecture Search (NAS)

```python
print("\n=== NEURAL ARCHITECTURE SEARCH ===")
print("""
Automatically design neural network architectures

SEARCH SPACE:
  - Number of layers
  - Layer types (conv, pool, dense)
  - Connections (skip connections)
  - Activation functions
  - Width of layers

SEARCH STRATEGIES:

1. REINFORCEMENT LEARNING:
   Controller network → generates architecture → trains → gets reward
   - NASNet, ENAS
   - Very expensive

2. EVOLUTIONARY:
   Population of architectures that evolve
   - AmoebaNet
   - Mutation: add/remove/change layers

3. DIFFERENTIABLE (DARTS):
   Relax discrete choices to continuous
   - Efficient gradient-based search
   - Orders of magnitude faster
   
4. WEIGHT SHARING:
   Share weights across candidate architectures
   - Train "supernet" once
   - Search by sampling subnetworks

COST COMPARISON:
  Method              GPU Hours
  Original NAS        22,400
  ENAS                0.5 (1,000x speedup)
  DARTS               1.5
  ProxylessNAS        8.3
""")

# Conceptual NAS search space
class NASSearchSpace:
    """Define what architectures we can search"""
    
    def __init__(self):
        self.operations = [
            'conv3x3', 'conv5x5', 'conv7x7',
            'maxpool3x3', 'avgpool3x3',
            'skip_connect', 'none'
        ]
        self.num_layers = range(2, 8)
        self.num_filters = [16, 32, 64, 128, 256]
    
    def sample_architecture(self):
        """Sample random architecture"""
        n_layers = np.random.choice(list(self.num_layers))
        architecture = []
        
        for i in range(n_layers):
            layer = {
                'operation': np.random.choice(self.operations),
                'filters': np.random.choice(self.num_filters)
            }
            architecture.append(layer)
        
        return architecture
    
    def describe(self, arch):
        """Describe architecture"""
        desc = []
        for i, layer in enumerate(arch):
            desc.append(f"  Layer {i}: {layer['operation']} ({layer['filters']} filters)")
        return '\n'.join(desc)

search_space = NASSearchSpace()
sample_arch = search_space.sample_architecture()

print("Sample architecture from search space:")
print(search_space.describe(sample_arch))
```

## Efficient NAS: DARTS

```python
print("\n=== DARTS: DIFFERENTIABLE NAS ===")
print("""
Key idea: Make architecture search differentiable

TRADITIONAL NAS:
  - Discrete choices (this op or that op)
  - Need RL or evolution
  - Very expensive

DARTS:
  - Mix all operations with learned weights
  - output = Σ α_i × op_i(input)
  - Learn α through gradient descent
  - After search: pick ops with highest α

ALGORITHM:
1. Create "mixed" operations with weights α
2. Alternating optimization:
   a. Update network weights w (standard training)
   b. Update architecture weights α (validation loss)
3. Discretize: keep operations with highest α

MUCH FASTER:
  - Single GPU, ~1 day
  - vs. thousands of GPU hours
""")

# Conceptual DARTS mixed operation
class MixedOperation:
    """Weighted mixture of candidate operations"""
    
    def __init__(self, operations):
        self.operations = operations
        # Architecture weights (learned)
        self.alphas = np.random.randn(len(operations))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def forward(self, x):
        """Mix all operations weighted by softmax(alpha)"""
        weights = self.softmax(self.alphas)
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        return output
    
    def best_operation(self):
        """Return operation with highest weight"""
        weights = self.softmax(self.alphas)
        best_idx = np.argmax(weights)
        return best_idx, weights[best_idx]

# Demo
operations = [
    lambda x: x * 1.0,  # identity
    lambda x: x * 2.0,  # scale
    lambda x: x ** 2,   # square
]
mixed_op = MixedOperation(operations)

print("DARTS mixed operation weights:")
print(f"  Alphas: {mixed_op.alphas.round(3)}")
print(f"  Softmax weights: {mixed_op.softmax(mixed_op.alphas).round(3)}")
best_idx, best_weight = mixed_op.best_operation()
print(f"  Best operation: index {best_idx} with weight {best_weight:.3f}")
```

## When to Use AutoML

```python
print("\n=== WHEN TO USE AUTOML ===")
print("""
GOOD USE CASES:
  ✓ Baseline establishment (quick benchmark)
  ✓ Limited ML expertise
  ✓ Many similar problems
  ✓ Standard ML tasks
  ✓ Hyperparameter tuning at scale

POOR USE CASES:
  ✗ Novel problem types
  ✗ Strict interpretability needs
  ✗ Extreme resource constraints
  ✗ Need for customization
  ✗ Real-time requirements

BEST PRACTICES:
1. Start with AutoML for baseline
2. Analyze what it found
3. Use insights to guide manual work
4. Compare AutoML vs manual
5. Consider hybrid approach

COMMON PITFALLS:
  - Overfitting to validation set
  - Ignoring compute costs
  - Black-box trust issues
  - Data leakage in pipeline
""")
```

## Key Points

- **AutoML**: Automates model selection and hyperparameter tuning
- **HPO methods**: Grid, random, Bayesian, evolutionary, early stopping
- **Frameworks**: Auto-sklearn, TPOT, H2O, cloud services
- **NAS**: Automated neural network design
- **DARTS**: Differentiable NAS for efficiency
- **Use wisely**: Great for baselines, not always for production

## Reflection Questions

1. When should you trust AutoML results vs. manual tuning?
2. What are the risks of using AutoML without understanding the process?
3. How do you validate that AutoML found a robust solution?
