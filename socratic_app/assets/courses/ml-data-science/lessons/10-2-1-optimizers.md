# Optimizers

## Introduction

Optimizers determine how neural network weights are updated during training. Beyond basic gradient descent, modern optimizers adapt learning rates and use momentum to speed up convergence.

## Gradient Descent Review

```python
import numpy as np
import pandas as pd

print("=== GRADIENT DESCENT ===")
print("""
Basic gradient descent update:
  w = w - learning_rate × gradient

Problems:
  1. Same learning rate for all parameters
  2. Can oscillate in ravines (elongated loss surfaces)
  3. May get stuck in local minima
  4. Learning rate is critical but hard to choose

Modern optimizers address these issues.
""")

# Create a simple loss function (2D for visualization)
def loss_function(w1, w2):
    """Bowl-shaped loss function"""
    return w1**2 + 10*w2**2  # Elongated in w2 direction

def gradient(w1, w2):
    """Gradient of loss"""
    return np.array([2*w1, 20*w2])

# Starting point
w = np.array([5.0, 1.0])
print(f"Starting point: w = {w}")
print(f"Initial loss: {loss_function(*w):.2f}")
```

## Vanilla Gradient Descent

```python
print("\n=== VANILLA GRADIENT DESCENT ===")

def vanilla_gd(w_init, lr, n_steps):
    """Standard gradient descent"""
    w = w_init.copy()
    history = [w.copy()]
    
    for _ in range(n_steps):
        grad = gradient(*w)
        w = w - lr * grad
        history.append(w.copy())
    
    return np.array(history)

lr = 0.05
history_vanilla = vanilla_gd(np.array([5.0, 1.0]), lr, 20)

print(f"Learning rate: {lr}")
print("Progress:")
for i in [0, 5, 10, 20]:
    w = history_vanilla[i]
    print(f"  Step {i:2d}: w = [{w[0]:6.3f}, {w[1]:6.3f}], loss = {loss_function(*w):.4f}")
```

## Momentum

```python
print("\n=== MOMENTUM ===")
print("""
Add velocity to smooth out oscillations:

v = momentum × v - lr × gradient
w = w + v

Intuition: Ball rolling down hill accumulates velocity
  - Accelerates in consistent direction
  - Dampens oscillations in inconsistent direction

Typical momentum: 0.9
""")

def momentum_gd(w_init, lr, momentum, n_steps):
    """Gradient descent with momentum"""
    w = w_init.copy()
    v = np.zeros_like(w)
    history = [w.copy()]
    
    for _ in range(n_steps):
        grad = gradient(*w)
        v = momentum * v - lr * grad
        w = w + v
        history.append(w.copy())
    
    return np.array(history)

history_momentum = momentum_gd(np.array([5.0, 1.0]), lr=0.05, momentum=0.9, n_steps=20)

print("Momentum comparison:")
print(f"{'Step':<6} {'Vanilla w':<20} {'Momentum w':<20}")
for i in [0, 5, 10, 20]:
    v = history_vanilla[i]
    m = history_momentum[i]
    print(f"{i:<6} [{v[0]:6.3f}, {v[1]:6.3f}]    [{m[0]:6.3f}, {m[1]:6.3f}]")
```

## AdaGrad

```python
print("\n=== ADAGRAD ===")
print("""
Adapt learning rate per parameter based on history:

accumulator += gradient²
w = w - lr × gradient / sqrt(accumulator + ε)

Effect:
  - Parameters with large gradients: smaller effective lr
  - Parameters with small gradients: larger effective lr
  - Good for sparse features

Problem: Learning rate always decreases (accumulator grows)
""")

def adagrad(w_init, lr, n_steps, eps=1e-8):
    """AdaGrad optimizer"""
    w = w_init.copy()
    accumulator = np.zeros_like(w)
    history = [w.copy()]
    
    for _ in range(n_steps):
        grad = gradient(*w)
        accumulator += grad ** 2
        w = w - lr * grad / (np.sqrt(accumulator) + eps)
        history.append(w.copy())
    
    return np.array(history)

history_adagrad = adagrad(np.array([5.0, 1.0]), lr=1.0, n_steps=20)

print("AdaGrad (note different learning rate):")
for i in [0, 5, 10, 20]:
    w = history_adagrad[i]
    print(f"  Step {i:2d}: w = [{w[0]:6.3f}, {w[1]:6.3f}], loss = {loss_function(*w):.4f}")
```

## RMSprop

```python
print("\n=== RMSprop ===")
print("""
Fix AdaGrad's diminishing learning rate:

accumulator = decay × accumulator + (1-decay) × gradient²
w = w - lr × gradient / sqrt(accumulator + ε)

Exponential moving average instead of sum.
Only considers recent gradients.

Typical decay: 0.9
""")

def rmsprop(w_init, lr, decay, n_steps, eps=1e-8):
    """RMSprop optimizer"""
    w = w_init.copy()
    accumulator = np.zeros_like(w)
    history = [w.copy()]
    
    for _ in range(n_steps):
        grad = gradient(*w)
        accumulator = decay * accumulator + (1 - decay) * grad ** 2
        w = w - lr * grad / (np.sqrt(accumulator) + eps)
        history.append(w.copy())
    
    return np.array(history)

history_rmsprop = rmsprop(np.array([5.0, 1.0]), lr=0.5, decay=0.9, n_steps=20)

print("RMSprop:")
for i in [0, 5, 10, 20]:
    w = history_rmsprop[i]
    print(f"  Step {i:2d}: w = [{w[0]:6.3f}, {w[1]:6.3f}], loss = {loss_function(*w):.4f}")
```

## Adam

```python
print("\n=== ADAM ===")
print("""
ADAM = Momentum + RMSprop + Bias correction

m = β₁ × m + (1-β₁) × gradient        # Momentum
v = β₂ × v + (1-β₂) × gradient²       # RMSprop

# Bias correction (important early in training)
m_hat = m / (1 - β₁^t)
v_hat = v / (1 - β₂^t)

w = w - lr × m_hat / sqrt(v_hat + ε)

Default hyperparameters:
  - lr = 0.001
  - β₁ = 0.9 (momentum)
  - β₂ = 0.999 (RMSprop decay)
  - ε = 1e-8

Adam is the default choice for many tasks!
""")

def adam(w_init, lr, beta1, beta2, n_steps, eps=1e-8):
    """Adam optimizer"""
    w = w_init.copy()
    m = np.zeros_like(w)  # First moment
    v = np.zeros_like(w)  # Second moment
    history = [w.copy()]
    
    for t in range(1, n_steps + 1):
        grad = gradient(*w)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update weights
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w.copy())
    
    return np.array(history)

history_adam = adam(np.array([5.0, 1.0]), lr=0.5, beta1=0.9, beta2=0.999, n_steps=20)

print("Adam:")
for i in [0, 5, 10, 20]:
    w = history_adam[i]
    print(f"  Step {i:2d}: w = [{w[0]:6.3f}, {w[1]:6.3f}], loss = {loss_function(*w):.4f}")
```

## Comparing Optimizers

```python
print("\n=== OPTIMIZER COMPARISON ===")

results = {
    'Vanilla GD': history_vanilla[-1],
    'Momentum': history_momentum[-1],
    'AdaGrad': history_adagrad[-1],
    'RMSprop': history_rmsprop[-1],
    'Adam': history_adam[-1]
}

print(f"{'Optimizer':<15} {'Final w':<25} {'Final Loss':<15}")
print("-" * 55)
for name, w in results.items():
    loss = loss_function(*w)
    print(f"{name:<15} [{w[0]:7.4f}, {w[1]:7.4f}]    {loss:.6f}")

print("""
Key observations:
  - Momentum accelerates convergence
  - Adaptive methods handle different scales
  - Adam combines benefits of both
""")
```

## Learning Rate Scheduling

```python
print("\n=== LEARNING RATE SCHEDULING ===")
print("""
Change learning rate during training:

1. STEP DECAY:
   lr = lr × drop_factor every N epochs
   
2. EXPONENTIAL DECAY:
   lr = lr₀ × decay^epoch
   
3. COSINE ANNEALING:
   lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))

4. WARMUP:
   - Start with small lr
   - Gradually increase
   - Then decay
   
5. REDUCE ON PLATEAU:
   - Reduce lr when loss stops improving
""")

def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
    """Step decay schedule"""
    return initial_lr * (drop_rate ** (epoch // epochs_drop))

def cosine_annealing(epoch, total_epochs, lr_max=0.1, lr_min=0.001):
    """Cosine annealing schedule"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))

print("Learning rate schedules over 50 epochs:")
print(f"{'Epoch':<8} {'Step Decay':<15} {'Cosine':<15}")
for epoch in [0, 10, 20, 30, 40, 50]:
    step_lr = step_decay(0.1, epoch)
    cos_lr = cosine_annealing(epoch, 50)
    print(f"{epoch:<8} {step_lr:<15.4f} {cos_lr:<15.4f}")
```

## Choosing an Optimizer

```python
print("\n=== CHOOSING AN OPTIMIZER ===")
print("""
Recommendations:

1. DEFAULT: Start with Adam
   - lr = 0.001
   - Works well in most cases

2. COMPUTER VISION: SGD + Momentum + LR schedule
   - Often achieves better final accuracy
   - Requires more tuning

3. NLP / TRANSFORMERS: Adam or AdamW
   - AdamW: Adam with weight decay fix

4. SPARSE DATA: AdaGrad or Adam
   - Adaptive learning rates help

5. FINE-TUNING: Lower learning rate
   - Pre-trained models need gentle updates

Common hyperparameters to tune:
  - Learning rate (most important!)
  - Batch size
  - Weight decay (L2 regularization)
""")
```

## Key Points

- **Vanilla GD**: Basic but can be slow, oscillates
- **Momentum**: Accumulates velocity, smooths oscillations
- **AdaGrad**: Per-parameter adaptive lr, good for sparse
- **RMSprop**: Fixes AdaGrad's diminishing lr
- **Adam**: Combines momentum and RMSprop, great default
- **LR scheduling**: Change learning rate during training
- **Adam is usually the best starting point**

## Reflection Questions

1. Why does momentum help in elongated loss landscapes?
2. When might SGD+momentum outperform Adam?
3. Why is bias correction important in the early steps of Adam?
