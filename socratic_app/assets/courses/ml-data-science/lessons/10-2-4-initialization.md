# Weight Initialization

## Introduction

How you initialize neural network weights matters enormously. Poor initialization can lead to vanishing or exploding gradients, making training impossible or extremely slow.

## Why Initialization Matters

```python
import numpy as np
import pandas as pd

print("=== WEIGHT INITIALIZATION ===")
print("""
PROBLEM: How to set initial weights?

Bad choices:
  - All zeros: All neurons learn the same thing (symmetry)
  - Too small: Signals vanish through layers
  - Too large: Signals explode, gradients unstable

Good initialization should:
  - Break symmetry (random)
  - Keep signal variance roughly constant across layers
  - Allow gradients to flow (not vanish/explode)
""")
```

## All Zeros Problem

```python
print("\n=== ALL ZEROS PROBLEM ===")

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, weights):
    """Forward pass through layers"""
    activations = [x]
    for W in weights:
        x = relu(np.dot(x, W))
        activations.append(x)
    return activations

# Zero initialization
np.random.seed(42)
n_layers = 3
layer_size = 4

zero_weights = [np.zeros((layer_size, layer_size)) for _ in range(n_layers)]

x = np.random.randn(1, layer_size)
activations = forward_pass(x, zero_weights)

print("With all zeros initialization:")
for i, a in enumerate(activations):
    print(f"  Layer {i}: {a[0].round(4)}")

print("""
All activations are zero!
All neurons compute the same thing.
No learning possible - symmetry not broken.
""")
```

## Too Small Initialization

```python
print("\n=== TOO SMALL INITIALIZATION ===")

# Very small random weights
small_weights = [np.random.randn(layer_size, layer_size) * 0.01 for _ in range(10)]

x = np.random.randn(1, layer_size)
activations = forward_pass(x, small_weights)

print("With small random initialization (std=0.01):")
print(f"  Input std: {x.std():.4f}")
for i, a in enumerate(activations):
    print(f"  Layer {i}: std = {a.std():.6f}")

print("""
Activations shrink to ~0 after a few layers!
Gradients will vanish - no learning in early layers.
""")
```

## Too Large Initialization

```python
print("\n=== TOO LARGE INITIALIZATION ===")

# Large random weights
large_weights = [np.random.randn(layer_size, layer_size) * 2.0 for _ in range(10)]

x = np.random.randn(1, layer_size)

print("With large random initialization (std=2.0):")
print(f"  Input std: {x.std():.4f}")

for i, W in enumerate(large_weights):
    x = np.dot(x, W)  # Skip ReLU to see explosion
    print(f"  Layer {i}: std = {x.std():.2e}")
    if x.std() > 1e10:
        print("  ... values exploding!")
        break

print("""
Activations explode after a few layers!
Gradients will explode or become NaN.
""")
```

## Xavier/Glorot Initialization

```python
print("\n=== XAVIER/GLOROT INITIALIZATION ===")
print("""
Goal: Keep variance constant across layers

For layer with n_in inputs and n_out outputs:

Xavier Uniform:
  W ~ Uniform(-limit, limit)
  limit = sqrt(6 / (n_in + n_out))

Xavier Normal:
  W ~ Normal(0, std)
  std = sqrt(2 / (n_in + n_out))

Best for: tanh, sigmoid activations
""")

def xavier_uniform(n_in, n_out):
    """Xavier/Glorot uniform initialization"""
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def xavier_normal(n_in, n_out):
    """Xavier/Glorot normal initialization"""
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std

# Test Xavier
n_in, n_out = 256, 256
W = xavier_normal(n_in, n_out)

print(f"Xavier normal for ({n_in}, {n_out}):")
print(f"  Expected std: {np.sqrt(2.0 / (n_in + n_out)):.4f}")
print(f"  Actual std: {W.std():.4f}")

# Deep network with Xavier
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

xavier_weights = [xavier_normal(64, 64) for _ in range(10)]
x = np.random.randn(1, 64)

print("\nDeep network with Xavier (sigmoid activations):")
for i, W in enumerate(xavier_weights):
    x = sigmoid(np.dot(x, W))
    if i % 2 == 0:
        print(f"  Layer {i}: mean={x.mean():.4f}, std={x.std():.4f}")
```

## He Initialization

```python
print("\n=== HE INITIALIZATION ===")
print("""
Designed for ReLU activations:

He Uniform:
  W ~ Uniform(-limit, limit)
  limit = sqrt(6 / n_in)

He Normal:
  W ~ Normal(0, std)
  std = sqrt(2 / n_in)

Why different from Xavier?
  - ReLU zeros out negative values
  - Reduces variance by ~half
  - He compensates with sqrt(2) factor

Best for: ReLU, Leaky ReLU, ELU
""")

def he_normal(n_in, n_out):
    """He/Kaiming normal initialization"""
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std

def he_uniform(n_in, n_out):
    """He/Kaiming uniform initialization"""
    limit = np.sqrt(6.0 / n_in)
    return np.random.uniform(-limit, limit, (n_in, n_out))

# Compare Xavier vs He for ReLU
print("ReLU deep network comparison:")

# Xavier with ReLU
xavier_weights = [xavier_normal(64, 64) for _ in range(10)]
x_xavier = np.random.randn(1, 64)

# He with ReLU
he_weights = [he_normal(64, 64) for _ in range(10)]
x_he = np.random.randn(1, 64)

print(f"{'Layer':<8} {'Xavier std':<15} {'He std':<15}")
print("-" * 38)
for i in range(10):
    x_xavier = relu(np.dot(x_xavier, xavier_weights[i]))
    x_he = relu(np.dot(x_he, he_weights[i]))
    print(f"{i:<8} {x_xavier.std():<15.4f} {x_he.std():<15.4f}")

print("\nHe maintains better variance with ReLU!")
```

## LeCun Initialization

```python
print("\n=== LECUN INITIALIZATION ===")
print("""
For SELU (self-normalizing) activation:

LeCun Normal:
  W ~ Normal(0, std)
  std = sqrt(1 / n_in)

LeCun Uniform:
  W ~ Uniform(-limit, limit)
  limit = sqrt(3 / n_in)

Similar to He but without the sqrt(2) factor.
""")

def lecun_normal(n_in, n_out):
    """LeCun normal initialization"""
    std = np.sqrt(1.0 / n_in)
    return np.random.randn(n_in, n_out) * std
```

## Initialization Summary

```python
print("\n=== INITIALIZATION SUMMARY ===")
print("""
Choosing initialization:

┌──────────────────┬────────────────────┬─────────────────┐
│ Activation       │ Initialization     │ Framework arg   │
├──────────────────┼────────────────────┼─────────────────┤
│ sigmoid, tanh    │ Xavier/Glorot      │ 'glorot_normal' │
│ ReLU, Leaky ReLU │ He/Kaiming         │ 'he_normal'     │
│ SELU             │ LeCun              │ 'lecun_normal'  │
│ Any              │ Orthogonal*        │ 'orthogonal'    │
└──────────────────┴────────────────────┴─────────────────┘

*Orthogonal: Good for RNNs, prevents vanishing gradients

PyTorch defaults:
  - Linear: Kaiming uniform
  - Conv2d: Kaiming uniform

Keras defaults:
  - Dense: Glorot uniform
  - Conv2D: Glorot uniform
""")

# Quick reference
print("\nQuick reference formulas:")
print(f"  Xavier std = sqrt(2 / (n_in + n_out))")
print(f"  He std = sqrt(2 / n_in)")
print(f"  LeCun std = sqrt(1 / n_in)")
```

## Practical Tips

```python
print("\n=== PRACTICAL TIPS ===")
print("""
1. Use framework defaults for common architectures
   - They usually set sensible initialization

2. Match initialization to activation
   - ReLU → He
   - Sigmoid/tanh → Xavier

3. Bias initialization
   - Usually zeros
   - Sometimes small positive for ReLU (ensures some neurons fire)

4. BatchNorm changes the game
   - Normalizes activations
   - Initialization less critical
   - But still use proper init

5. Transfer learning
   - Use pre-trained weights
   - Fine-tune last few layers

6. Signs of bad initialization
   - Loss stays flat (dead neurons)
   - Loss is NaN (exploding)
   - Very slow training
""")
```

## Key Points

- **Don't use all zeros**: Breaks symmetry requirement
- **Too small**: Signals vanish
- **Too large**: Signals explode
- **Xavier/Glorot**: For sigmoid, tanh
- **He/Kaiming**: For ReLU, Leaky ReLU
- **Match init to activation function**
- **Framework defaults are usually good**

## Reflection Questions

1. Why does ReLU need a different initialization than sigmoid?
2. How does batch normalization affect the importance of initialization?
3. What happens if all neurons start with identical weights?
