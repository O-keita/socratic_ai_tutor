# Continual and Lifelong Learning

## Introduction

Continual learning enables models to learn from a stream of data over time, acquiring new knowledge while retaining previously learned information. This is critical for deploying ML systems in dynamic environments.

## The Catastrophic Forgetting Problem

```python
import numpy as np
import pandas as pd

print("=== CATASTROPHIC FORGETTING ===")
print("""
PROBLEM: Neural networks forget old tasks when learning new ones

Traditional ML:
  Train on Dataset A → Good on A
  Then train on Dataset B → Good on B, BAD on A!

WHY IT HAPPENS:
  - Neural networks overwrite old weights
  - New task gradients interfere with old task representations
  - No explicit memory of past data

EXAMPLE:
  Task 1: Classify cats vs dogs
  Task 2: Classify birds vs fish
  
  After Task 2: Model forgets cats and dogs!

REAL-WORLD IMPACT:
  - Chatbots forget old conversations
  - Recommendation systems forget user preferences
  - Autonomous cars forget rare scenarios
""")

# Demo catastrophic forgetting
np.random.seed(42)

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    
    def forward(self, X):
        self.h = np.maximum(0, X @ self.W1)  # ReLU
        return self.h @ self.W2
    
    def train(self, X, y, epochs=100, lr=0.01):
        for _ in range(epochs):
            # Forward
            pred = self.forward(X)
            
            # Backward (simplified)
            grad2 = self.h.T @ (pred - y) / len(X)
            grad1 = X.T @ ((pred - y) @ self.W2.T * (self.h > 0)) / len(X)
            
            self.W1 -= lr * grad1
            self.W2 -= lr * grad2
    
    def evaluate(self, X, y):
        pred = self.forward(X)
        return np.mean((pred - y) ** 2)

# Create two tasks
# Task 1: Linear function
X1 = np.random.randn(100, 5)
y1 = X1[:, 0:1] + X1[:, 1:2]  # y = x0 + x1

# Task 2: Different linear function
X2 = np.random.randn(100, 5)
y2 = X2[:, 2:3] - X2[:, 3:4]  # y = x2 - x3

model = SimpleNN(5, 10, 1)

# Train on Task 1
model.train(X1, y1, epochs=200)
loss1_after_t1 = model.evaluate(X1, y1)
print(f"After Task 1: Loss on Task 1 = {loss1_after_t1:.4f}")

# Train on Task 2
model.train(X2, y2, epochs=200)
loss1_after_t2 = model.evaluate(X1, y1)
loss2_after_t2 = model.evaluate(X2, y2)

print(f"After Task 2: Loss on Task 1 = {loss1_after_t2:.4f} (Forgetting!)")
print(f"After Task 2: Loss on Task 2 = {loss2_after_t2:.4f}")
```

## Continual Learning Strategies

```python
print("\n=== CONTINUAL LEARNING APPROACHES ===")
print("""
1. REGULARIZATION-BASED:
   Add penalty for changing important weights
   - EWC (Elastic Weight Consolidation)
   - SI (Synaptic Intelligence)
   - LwF (Learning without Forgetting)

2. REPLAY-BASED:
   Store and replay old examples
   - Experience Replay
   - Generative Replay
   - Gradient Episodic Memory (GEM)

3. ARCHITECTURE-BASED:
   Allocate different capacity for different tasks
   - Progressive Neural Networks
   - PackNet
   - Dynamically Expandable Networks

4. OPTIMIZATION-BASED:
   Modify gradient updates
   - OGD (Orthogonal Gradient Descent)
   - A-GEM
   - GPM (Gradient Projection Memory)
""")
```

## Elastic Weight Consolidation (EWC)

```python
print("\n=== EWC ALGORITHM ===")
print("""
Key idea: Protect important weights for old tasks

IMPORTANCE = Fisher Information
  - How much does loss change when weight changes?
  - High Fisher = important for old task
  
LOSS = Standard Loss + λ Σ F_i (θ_i - θ*_i)²

Where:
  - F_i: Fisher information (importance) of weight i
  - θ*_i: Optimal weight after old task
  - θ_i: Current weight
  - λ: Regularization strength

ALGORITHM:
1. Train on Task 1, get θ*
2. Compute Fisher information F
3. Train on Task 2 with:
   L = L_task2 + λ/2 × Σ F_i × (θ_i - θ*_i)²

Important weights can't change much!
""")

class EWC:
    """Elastic Weight Consolidation"""
    
    def __init__(self, model, lambda_ewc=1000):
        self.lambda_ewc = lambda_ewc
        self.old_params = None
        self.fisher = None
    
    def compute_fisher(self, model, X, y, n_samples=100):
        """Estimate Fisher Information using gradients"""
        fisher = {}
        
        # Initialize
        fisher['W1'] = np.zeros_like(model.W1)
        fisher['W2'] = np.zeros_like(model.W2)
        
        # Sample-wise gradients squared
        for i in range(min(n_samples, len(X))):
            x_i = X[i:i+1]
            y_i = y[i:i+1]
            
            # Forward
            h = np.maximum(0, x_i @ model.W1)
            pred = h @ model.W2
            
            # Gradients
            grad2 = h.T @ (pred - y_i)
            grad1 = x_i.T @ ((pred - y_i) @ model.W2.T * (h > 0))
            
            fisher['W1'] += grad1 ** 2
            fisher['W2'] += grad2 ** 2
        
        # Normalize
        for key in fisher:
            fisher[key] /= n_samples
        
        return fisher
    
    def store_params(self, model):
        """Store parameters after training on task"""
        self.old_params = {
            'W1': model.W1.copy(),
            'W2': model.W2.copy()
        }
    
    def penalty(self, model):
        """Compute EWC penalty"""
        if self.old_params is None or self.fisher is None:
            return 0
        
        loss = 0
        loss += np.sum(self.fisher['W1'] * (model.W1 - self.old_params['W1']) ** 2)
        loss += np.sum(self.fisher['W2'] * (model.W2 - self.old_params['W2']) ** 2)
        
        return 0.5 * self.lambda_ewc * loss

print("EWC prevents catastrophic forgetting by protecting important weights")
```

## Experience Replay

```python
print("\n=== EXPERIENCE REPLAY ===")
print("""
Key idea: Store and replay examples from old tasks

SIMPLE REPLAY:
  - Store subset of old examples
  - Mix with new data when training
  - Simple and effective

RESERVOIR SAMPLING:
  - Fixed-size buffer
  - Uniform sampling over stream
  - No need to know total size

GENERATIVE REPLAY:
  - Train generative model on old data
  - Generate synthetic old examples
  - No need to store actual data
  - Privacy-preserving
""")

class ReplayBuffer:
    """Experience replay buffer with reservoir sampling"""
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0
    
    def add(self, x, y):
        """Add example using reservoir sampling"""
        self.n_seen += 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            # Reservoir sampling
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = (x, y)
    
    def sample(self, batch_size):
        """Sample batch from buffer"""
        indices = np.random.choice(len(self.buffer), 
                                   min(batch_size, len(self.buffer)), 
                                   replace=False)
        
        X = np.array([self.buffer[i][0] for i in indices])
        y = np.array([self.buffer[i][1] for i in indices])
        return X, y

# Demo
buffer = ReplayBuffer(capacity=100)

# Add examples from Task 1
for i in range(50):
    buffer.add(X1[i], y1[i])

print(f"Buffer size after Task 1: {len(buffer.buffer)}")

# Sample for replay during Task 2 training
X_replay, y_replay = buffer.sample(10)
print(f"Replay batch shape: {X_replay.shape}")
```

## Progressive Neural Networks

```python
print("\n=== PROGRESSIVE NETWORKS ===")
print("""
Key idea: Grow network for new tasks, freeze old columns

ARCHITECTURE:
Task 1:  Task 2:  Task 3:
  □        □        □
  │        │╲       │╲╲
  □        □ ←──── □ ←──
  │        │╲       │╲╲
  □        □        □

Each task gets a new column
Old columns frozen
Lateral connections from old to new

BENEFITS:
  ✓ No forgetting (old weights frozen)
  ✓ Transfer via lateral connections
  ✓ Can learn task-specific features

DRAWBACKS:
  ✗ Network grows linearly with tasks
  ✗ High memory for many tasks
  ✗ Need to know task boundaries
""")

class ProgressiveNetwork:
    """Simplified progressive neural network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.columns = []  # Each column is for a task
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def add_column(self):
        """Add new column for new task"""
        column_id = len(self.columns)
        
        # New column weights
        column = {
            'W1': np.random.randn(self.input_dim, self.hidden_dim) * 0.1,
            'W2': np.random.randn(self.hidden_dim, self.output_dim) * 0.1,
            'lateral': []  # Lateral connections from previous columns
        }
        
        # Add lateral connections from all previous columns
        for prev_col in range(column_id):
            lateral = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
            column['lateral'].append(lateral)
        
        self.columns.append(column)
        print(f"Added column {column_id} with {len(column['lateral'])} lateral connections")
        
        return column_id
    
    def forward(self, X, task_id):
        """Forward pass for specific task"""
        column = self.columns[task_id]
        
        # First layer
        h = np.maximum(0, X @ column['W1'])
        
        # Add lateral connections from previous columns
        for prev_id, lateral in enumerate(column['lateral']):
            prev_h = np.maximum(0, X @ self.columns[prev_id]['W1'])
            h = h + prev_h @ lateral
        
        # Output
        return h @ column['W2']

# Demo
prog_net = ProgressiveNetwork(5, 10, 1)

# Add columns for 3 tasks
for task in range(3):
    prog_net.add_column()

print(f"\nNetwork has {len(prog_net.columns)} columns")
```

## Continual Learning Benchmarks

```python
print("\n=== CL BENCHMARKS ===")
print("""
Standard benchmarks for evaluation:

1. PERMUTED MNIST:
   - Same digits, different pixel permutation per task
   - Tests memory capacity
   
2. SPLIT MNIST/CIFAR:
   - Different classes per task
   - Task 1: 0-1, Task 2: 2-3, etc.
   - Tests class-incremental learning

3. CORE50:
   - Object recognition
   - 50 objects, 11 sessions
   - Different backgrounds/contexts

4. CLEAR:
   - Decade of visual data
   - Natural temporal shift
   - Realistic distribution shift

METRICS:

1. Average Accuracy:
   A = (1/T) Σ a_T,i
   After T tasks, average accuracy on all tasks

2. Forgetting:
   F = (1/T-1) Σ (max_j a_j,i - a_T,i)
   How much was forgotten?

3. Forward Transfer:
   FT = (1/T-1) Σ (a_i-1,i - a_0,i)
   Did old tasks help new ones?

4. Backward Transfer:
   BT = (1/T-1) Σ (a_T,i - a_i,i)
   Did new tasks help old ones?
""")

def compute_cl_metrics(accuracy_matrix):
    """Compute CL metrics from accuracy matrix
    
    accuracy_matrix[i,j] = accuracy on task j after training on task i
    """
    T = len(accuracy_matrix)
    
    # Average accuracy after all tasks
    avg_accuracy = np.mean(accuracy_matrix[-1])
    
    # Forgetting
    forgetting = 0
    for j in range(T - 1):
        max_acc = max(accuracy_matrix[i, j] for i in range(j, T))
        final_acc = accuracy_matrix[-1, j]
        forgetting += max_acc - final_acc
    forgetting /= (T - 1) if T > 1 else 1
    
    # Backward transfer
    bwt = 0
    for j in range(T - 1):
        bwt += accuracy_matrix[-1, j] - accuracy_matrix[j, j]
    bwt /= (T - 1) if T > 1 else 1
    
    return {
        'avg_accuracy': avg_accuracy,
        'forgetting': forgetting,
        'backward_transfer': bwt
    }

# Example accuracy matrix
acc_matrix = np.array([
    [0.95, 0.00, 0.00],  # After task 1
    [0.80, 0.92, 0.00],  # After task 2
    [0.70, 0.85, 0.90],  # After task 3
])

metrics = compute_cl_metrics(acc_matrix)
print("CL Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")
```

## Key Points

- **Catastrophic forgetting**: NNs forget old tasks when learning new
- **Regularization**: EWC protects important weights
- **Replay**: Store and replay old examples
- **Architecture**: Grow network for new tasks (Progressive)
- **Metrics**: Average accuracy, forgetting, transfer
- **Benchmarks**: Permuted MNIST, Split CIFAR, CORe50

## Reflection Questions

1. When is catastrophic forgetting acceptable vs. critical to prevent?
2. What are the privacy implications of experience replay?
3. How do you handle the unknown number of future tasks?
