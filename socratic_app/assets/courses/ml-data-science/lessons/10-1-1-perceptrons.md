# Neural Network Basics: Perceptrons and Neurons

## Introduction

Neural networks are inspired by the human brain. Understanding the basic building block—the artificial neuron—is essential for grasping how deep learning works.

## Core Concepts

### The Perceptron

The perceptron is the simplest neural network unit:

$$output = f(\sum_{i=1}^{n} w_i x_i + b)$$

Where:
- $x_i$ are inputs
- $w_i$ are weights
- $b$ is bias
- $f$ is the activation function

### From Perceptron to Neural Network

```python
import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        # Weighted sum
        z = np.dot(inputs, self.weights) + self.bias
        # Activation
        return self.sigmoid(z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
```

### Multi-Layer Networks

```python
import torch
import torch.nn as nn

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model
model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
```

### Layer Types

**Fully Connected (Dense) Layer:**
Every neuron connects to all neurons in the previous layer.

```python
# PyTorch
layer = nn.Linear(in_features=100, out_features=50)

# TensorFlow/Keras
layer = Dense(units=50, activation='relu')
```

### Weight Initialization

Proper initialization is crucial:

```python
# Xavier/Glorot initialization (good for tanh, sigmoid)
nn.init.xavier_uniform_(layer.weight)

# He initialization (good for ReLU)
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

# In Keras
layer = Dense(64, kernel_initializer='he_normal')
```

### Forward Pass

```python
# Input flows through network
x = torch.randn(32, 10)  # batch of 32, 10 features

# Forward pass
output = model(x)
print(output.shape)  # (32, 2)
```

---

## Key Points

- Neurons compute weighted sum of inputs plus bias, then apply activation
- Multiple layers enable learning complex patterns
- Weights and biases are the learnable parameters
- Proper initialization affects training stability
- Forward pass transforms input through successive layers

---

## Reflection Questions

1. **Think**: Why is the bias term important? What would happen if we only had weights?

2. **Consider**: How does adding more hidden layers change what a network can learn? What's the tradeoff?

3. **Explore**: Why does weight initialization matter? What could go wrong with random large values?
