# Backpropagation

## Introduction

Backpropagation is the algorithm that enables neural networks to learn by calculating gradients of the loss function with respect to each weight.

## The Learning Problem

```python
import numpy as np
import pandas as pd

print("=== BACKPROPAGATION ===")
print("""
GOAL: Find weights that minimize loss

Forward pass: Input → Predictions → Loss
Backward pass: Loss → Gradients → Weight updates

Key question: How does each weight affect the loss?
  
Answer: Compute ∂Loss/∂w for every weight w

Then update: w_new = w_old - learning_rate × ∂Loss/∂w

This is GRADIENT DESCENT applied to neural networks.
""")
```

## Chain Rule Foundation

```python
print("\n=== CHAIN RULE ===")
print("""
Backpropagation uses the CHAIN RULE from calculus:

If z = f(g(x)), then:
  dz/dx = dz/dg × dg/dx

Neural network example:
  loss = L(σ(w×x + b))
  
To find ∂loss/∂w:
  ∂loss/∂w = ∂loss/∂σ × ∂σ/∂z × ∂z/∂w
  
Where z = w×x + b

We compute gradients layer by layer, going backward.
""")

# Chain rule example
def f(x):
    return x ** 2

def g(x):
    return 3 * x + 1

# z = f(g(x)) = (3x + 1)^2
# dz/dx = 2(3x + 1) × 3 = 6(3x + 1)

x = 2
g_x = g(x)           # 3*2 + 1 = 7
f_g_x = f(g_x)       # 7^2 = 49

# Derivatives
df_dg = 2 * g_x      # f'(g) = 2g = 14
dg_dx = 3            # g'(x) = 3
dz_dx = df_dg * dg_dx  # Chain rule = 42

print(f"z = f(g(x)) = (3x + 1)² at x={x}")
print(f"g(x) = {g_x}")
print(f"z = {f_g_x}")
print(f"\nChain rule derivatives:")
print(f"  df/dg = {df_dg}")
print(f"  dg/dx = {dg_dx}")
print(f"  dz/dx = df/dg × dg/dx = {dz_dx}")
```

## Simple Neural Network

```python
print("\n=== SIMPLE NEURAL NETWORK ===")

def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Derivative of sigmoid"""
    s = sigmoid(z)
    return s * (1 - s)

def mse_loss(y_true, y_pred):
    """Mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

# Network: 2 inputs → 2 hidden → 1 output
np.random.seed(42)

# Initialize weights
W1 = np.random.randn(2, 2) * 0.5  # Input to hidden
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.5  # Hidden to output
b2 = np.zeros((1, 1))

print("Network architecture: 2 → 2 → 1")
print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")
```

## Forward Pass

```python
print("\n=== FORWARD PASS ===")

# Sample data
X = np.array([[0.5, 0.8]])  # 1 sample, 2 features
y = np.array([[1.0]])        # Target

# Forward pass
z1 = np.dot(X, W1) + b1      # Linear transformation
a1 = sigmoid(z1)              # Activation
z2 = np.dot(a1, W2) + b2     # Linear transformation
y_pred = sigmoid(z2)          # Output

loss = mse_loss(y, y_pred)

print("Forward pass:")
print(f"  Input X: {X}")
print(f"  z1 (linear): {z1.round(4)}")
print(f"  a1 (sigmoid): {a1.round(4)}")
print(f"  z2 (linear): {z2.round(4)}")
print(f"  y_pred: {y_pred.round(4)}")
print(f"  y_true: {y}")
print(f"  Loss: {loss:.4f}")
```

## Backward Pass

```python
print("\n=== BACKWARD PASS ===")
print("""
Compute gradients going backward:

1. ∂L/∂y_pred (loss gradient)
2. ∂L/∂z2 (through output activation)
3. ∂L/∂W2, ∂L/∂b2 (output layer weights)
4. ∂L/∂a1 (gradient flows backward)
5. ∂L/∂z1 (through hidden activation)
6. ∂L/∂W1, ∂L/∂b1 (hidden layer weights)
""")

# Backward pass
# Step 1: Loss gradient
dL_dy_pred = 2 * (y_pred - y) / y.size  # MSE derivative

# Step 2: Through output sigmoid
dy_pred_dz2 = sigmoid_derivative(z2)
dL_dz2 = dL_dy_pred * dy_pred_dz2

# Step 3: Output layer gradients
dL_dW2 = np.dot(a1.T, dL_dz2)
dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

# Step 4: Gradient flows to hidden layer
dL_da1 = np.dot(dL_dz2, W2.T)

# Step 5: Through hidden sigmoid
da1_dz1 = sigmoid_derivative(z1)
dL_dz1 = dL_da1 * da1_dz1

# Step 6: Hidden layer gradients
dL_dW1 = np.dot(X.T, dL_dz1)
dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

print("Gradients computed:")
print(f"  dL/dW2 shape: {dL_dW2.shape}, values: {dL_dW2.flatten().round(4)}")
print(f"  dL/db2 shape: {dL_db2.shape}, values: {dL_db2.flatten().round(4)}")
print(f"  dL/dW1 shape: {dL_dW1.shape}")
print(f"  dL/db1 shape: {dL_db1.shape}")
```

## Weight Update

```python
print("\n=== WEIGHT UPDATE ===")

learning_rate = 0.5

# Store old weights
W1_old = W1.copy()
W2_old = W2.copy()

# Update weights
W1 = W1 - learning_rate * dL_dW1
b1 = b1 - learning_rate * dL_db1
W2 = W2 - learning_rate * dL_dW2
b2 = b2 - learning_rate * dL_db2

print(f"Learning rate: {learning_rate}")
print(f"\nW2 update:")
print(f"  Old: {W2_old.flatten().round(4)}")
print(f"  Gradient: {dL_dW2.flatten().round(4)}")
print(f"  New: {W2.flatten().round(4)}")

# Verify loss decreased
z1_new = np.dot(X, W1) + b1
a1_new = sigmoid(z1_new)
z2_new = np.dot(a1_new, W2) + b2
y_pred_new = sigmoid(z2_new)
loss_new = mse_loss(y, y_pred_new)

print(f"\nLoss: {loss:.4f} → {loss_new:.4f}")
print(f"Loss decreased by: {loss - loss_new:.4f}")
```

## Complete Training Loop

```python
print("\n=== COMPLETE TRAINING ===")

# Reset weights
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))

# Training data
X_train = np.array([[0.5, 0.8], [0.2, 0.9], [0.8, 0.1], [0.1, 0.2]])
y_train = np.array([[1], [1], [0], [0]])

learning_rate = 1.0
n_epochs = 100

losses = []
for epoch in range(n_epochs):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    
    loss = mse_loss(y_train, y_pred)
    losses.append(loss)
    
    # Backward pass
    m = X_train.shape[0]
    dL_dy_pred = 2 * (y_pred - y_train) / m
    dL_dz2 = dL_dy_pred * sigmoid_derivative(z2)
    dL_dW2 = np.dot(a1.T, dL_dz2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    dL_da1 = np.dot(dL_dz2, W2.T)
    dL_dz1 = dL_da1 * sigmoid_derivative(z1)
    dL_dW1 = np.dot(X_train.T, dL_dz1)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    
    # Update weights
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")

print(f"\nFinal predictions vs targets:")
for i in range(len(y_train)):
    print(f"  {y_pred[i][0]:.3f} vs {y_train[i][0]}")
```

## Computational Graph

```python
print("\n=== COMPUTATIONAL GRAPH ===")
print("""
Neural network as a graph of operations:

     X
     ↓
    W1×X + b1 = z1
     ↓
   sigmoid(z1) = a1
     ↓
    W2×a1 + b2 = z2
     ↓
   sigmoid(z2) = y_pred
     ↓
    MSE(y_pred, y) = Loss

Backprop: Reverse the arrows, multiply local gradients

Modern frameworks (PyTorch, TensorFlow):
  - Build computational graph automatically
  - Compute gradients via autograd
  - loss.backward() computes all gradients!
""")
```

## Key Points

- **Backpropagation**: Algorithm to compute gradients in neural networks
- **Chain rule**: Foundation for computing gradients through compositions
- **Forward pass**: Compute predictions and loss
- **Backward pass**: Compute gradients layer by layer
- **Weight update**: Subtract learning_rate × gradient from weights
- **Computational graph**: View network as operations, reverse for gradients

## Reflection Questions

1. Why do we compute gradients starting from the output layer going backward?
2. What happens if the learning rate is too large or too small?
3. How does the sigmoid derivative affect gradient flow (hint: vanishing gradients)?
