# Regularization in Neural Networks

## Introduction

Regularization prevents neural networks from overfitting by constraining model complexity. Multiple techniques exist for neural networks beyond traditional L1/L2 regularization.

## The Overfitting Problem

```python
import numpy as np
import pandas as pd

print("=== REGULARIZATION IN NEURAL NETWORKS ===")
print("""
OVERFITTING in neural networks:
  - Model memorizes training data
  - Poor generalization to new data
  - Training loss ↓, validation loss ↑

Neural networks are especially prone because:
  - Many parameters (millions+)
  - High capacity to memorize
  - Complex decision boundaries

Regularization techniques:
  1. L2 regularization (weight decay)
  2. Dropout
  3. Batch normalization
  4. Early stopping
  5. Data augmentation
""")
```

## L2 Regularization (Weight Decay)

```python
print("\n=== L2 REGULARIZATION ===")
print("""
Add penalty for large weights:

Loss_total = Loss_data + λ × Σ w²

Effect:
  - Weights stay small
  - Simpler model
  - Reduces overfitting

Also called "weight decay" because:
  w = w - lr × (gradient + λ × w)
  = w × (1 - lr × λ) - lr × gradient
  
Weights "decay" toward zero each update.
""")

# Demonstrate L2 effect
def compute_loss_with_l2(data_loss, weights, lambda_reg):
    """Compute total loss with L2 regularization"""
    l2_penalty = lambda_reg * np.sum(weights ** 2)
    return data_loss + l2_penalty

# Example
weights = np.array([0.5, 1.2, -0.8, 2.1])
data_loss = 0.3

for lambda_reg in [0, 0.01, 0.1]:
    total_loss = compute_loss_with_l2(data_loss, weights, lambda_reg)
    print(f"λ={lambda_reg:.2f}: Total loss = {total_loss:.4f}")

print("\nLarger λ → stronger regularization → simpler model")
```

## Dropout

```python
print("\n=== DROPOUT ===")
print("""
Randomly "drop" neurons during training:

During training:
  - Each neuron dropped with probability p
  - Remaining neurons scaled by 1/(1-p)
  
During inference:
  - All neurons active
  - No scaling needed (already normalized)

Intuition:
  - Prevents co-adaptation of neurons
  - Like training ensemble of networks
  - Each forward pass = different subnetwork
""")

def dropout(x, drop_rate, training=True):
    """Apply dropout to layer output"""
    if not training:
        return x  # No dropout during inference
    
    # Create mask
    mask = np.random.binomial(1, 1 - drop_rate, size=x.shape)
    
    # Apply mask and scale
    return x * mask / (1 - drop_rate)

np.random.seed(42)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

print(f"Original: {x}")
print(f"With 50% dropout (training):")
for i in range(3):
    dropped = dropout(x, drop_rate=0.5, training=True)
    print(f"  Trial {i+1}: {dropped.round(2)}")

print(f"\nWith dropout (inference): {dropout(x, 0.5, training=False)}")
```

## Dropout Implementation

```python
print("\n=== DROPOUT IN A NETWORK ===")

class DropoutLayer:
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward(self, x, training=True):
        if not training or self.drop_rate == 0:
            return x
        
        # Create and store mask for backprop
        self.mask = np.random.binomial(1, 1 - self.drop_rate, size=x.shape)
        return x * self.mask / (1 - self.drop_rate)
    
    def backward(self, grad_output):
        # Gradient only flows through non-dropped neurons
        return grad_output * self.mask / (1 - self.drop_rate)

print("""
Typical dropout rates:
  - Input layer: 0.1-0.2 (less dropout)
  - Hidden layers: 0.3-0.5
  - Output layer: No dropout
  
Higher dropout → more regularization but may underfit
""")
```

## Batch Normalization

```python
print("\n=== BATCH NORMALIZATION ===")
print("""
Normalize activations within each mini-batch:

1. Compute batch mean: μ = (1/m) Σ x
2. Compute batch variance: σ² = (1/m) Σ (x - μ)²
3. Normalize: x̂ = (x - μ) / sqrt(σ² + ε)
4. Scale and shift: y = γ × x̂ + β

Where γ and β are learnable parameters.

Benefits:
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Acts as regularization
  - Faster training
""")

class BatchNorm:
    def __init__(self, n_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        
        # Running stats for inference
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
    
    def forward(self, x, training=True):
        if training:
            # Compute batch statistics
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running stats
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example
np.random.seed(42)
batch = np.random.randn(4, 3) * 10 + 50  # Unnormalized data

bn = BatchNorm(3)
normalized = bn.forward(batch, training=True)

print("Before BatchNorm:")
print(f"  Mean: {batch.mean(axis=0).round(2)}")
print(f"  Std: {batch.std(axis=0).round(2)}")

print("\nAfter BatchNorm:")
print(f"  Mean: {normalized.mean(axis=0).round(4)}")
print(f"  Std: {normalized.std(axis=0).round(4)}")
```

## Early Stopping

```python
print("\n=== EARLY STOPPING ===")
print("""
Stop training when validation loss stops improving:

1. Train and monitor validation loss
2. Save model when validation loss improves
3. If no improvement for N epochs, stop
4. Use saved best model

Hyperparameters:
  - patience: How many epochs to wait
  - min_delta: Minimum improvement to count
""")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def check(self, val_loss, model_weights):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model_weights.copy()
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False

# Simulate training with early stopping
np.random.seed(42)
val_losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.45, 0.46, 0.47, 0.48, 0.50]

early_stop = EarlyStopping(patience=3)

print("Epoch | Val Loss | Action")
print("-" * 35)
for epoch, loss in enumerate(val_losses):
    weights = np.array([epoch])  # Dummy weights
    should_stop = early_stop.check(loss, weights)
    status = "Stop!" if should_stop else ("New best" if early_stop.counter == 0 else f"Patience {early_stop.counter}/{early_stop.patience}")
    print(f"  {epoch:2d}  |  {loss:.3f}   | {status}")
    if should_stop:
        break

print(f"\nBest model from epoch {early_stop.best_weights[0]}")
```

## Combining Regularization Techniques

```python
print("\n=== COMBINING TECHNIQUES ===")
print("""
Typical modern network:

Input → Dense → BatchNorm → ReLU → Dropout
      → Dense → BatchNorm → ReLU → Dropout
      → Output

Combination tips:
  - BatchNorm before activation
  - Dropout after activation
  - L2 regularization on all layers
  - Early stopping always

Reduce regularization if:
  - Training loss is high
  - Model underfits

Increase regularization if:
  - Train-val gap is large
  - Validation loss increases while train decreases
""")

print("""
Example architecture in Keras:

model = Sequential([
    Dense(128, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    
    Dense(64, kernel_regularizer=l2(0.01)),
    BatchNormalization(), 
    Activation('relu'),
    Dropout(0.3),
    
    Dense(n_classes, activation='softmax')
])
""")
```

## Key Points

- **L2 regularization**: Penalize large weights, simpler models
- **Dropout**: Randomly drop neurons, prevents co-adaptation
- **Batch normalization**: Normalize activations, faster training
- **Early stopping**: Stop when validation loss plateaus
- **Combine techniques** for best results
- **Monitor train-val gap** to diagnose over/underfitting

## Reflection Questions

1. Why does dropout work like training an ensemble of networks?
2. How does batch normalization act as regularization?
3. What signs tell you to increase vs decrease regularization strength?
