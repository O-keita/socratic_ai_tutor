# Activation Functions

## Introduction

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns that linear models cannot capture.

## Core Concepts

### Why Activation Functions?

Without non-linearity, stacking layers is pointless:

```python
# Linear: f(x) = ax + b
# Two linear layers: f(g(x)) = a(bx + c) + d = abx + (ac + d)
# Still linear! Can be replaced by single layer.
```

### Common Activation Functions

#### Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output range: (0, 1)
# Good for: binary classification output
# Problem: vanishing gradients, not zero-centered
```

#### Tanh
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```python
def tanh(x):
    return np.tanh(x)

# Output range: (-1, 1)
# Better than sigmoid: zero-centered
# Still has vanishing gradient problem
```

#### ReLU (Rectified Linear Unit)
$$ReLU(x) = \max(0, x)$$

```python
def relu(x):
    return np.maximum(0, x)

# Output range: [0, ∞)
# Fast to compute
# Most popular for hidden layers
# Problem: "dying ReLU" - neurons can stop learning
```

#### Leaky ReLU
$$LeakyReLU(x) = \max(\alpha x, x)$$

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Fixes dying ReLU problem
# Small gradient for negative values
```

#### Softmax
$$softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # numerical stability
    return exp_x / exp_x.sum()

# Output: probability distribution (sums to 1)
# Used for multi-class classification output
```

### Using Activations in PyTorch

```python
import torch.nn as nn

# As layers
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.Softmax(dim=1)
)

# As functions
import torch.nn.functional as F

x = F.relu(layer1(x))
x = F.leaky_relu(layer2(x), negative_slope=0.01)
```

### Choosing Activations

| Layer Type | Recommended Activation |
|------------|----------------------|
| Hidden layers | ReLU, Leaky ReLU |
| Binary output | Sigmoid |
| Multi-class output | Softmax |
| Regression output | None (Linear) |
| RNN hidden | Tanh |

---

## Key Points

- Activation functions add non-linearity to networks
- ReLU is standard for hidden layers (fast, effective)
- Sigmoid/Softmax for output layers in classification
- Vanishing gradients affect sigmoid/tanh in deep networks
- Choice of activation affects training dynamics

---

## Reflection Questions

1. **Think**: Why does ReLU work better than sigmoid for deep networks? What happens to gradients in deep sigmoid networks?

2. **Consider**: When might Leaky ReLU be preferred over standard ReLU? What problem does it solve?

3. **Explore**: Why does softmax use `exp(x - max(x))` instead of just `exp(x)`? What numerical issue does this prevent?
