# Introduction to CNNs

## Introduction

Convolutional Neural Networks (CNNs) are specialized neural networks designed to process grid-like data such as images. They automatically learn spatial hierarchies of features.

## Why CNNs for Images?

```python
import numpy as np
import pandas as pd

print("=== CONVOLUTIONAL NEURAL NETWORKS ===")
print("""
PROBLEM with fully connected layers for images:

Image: 224 × 224 × 3 = 150,528 pixels
First dense layer (1000 neurons): 150,528 × 1000 = 150 million parameters!

Issues:
  1. Too many parameters (overfitting)
  2. Ignores spatial structure
  3. Not translation invariant

CNN SOLUTION:
  - Local connections (convolutions)
  - Parameter sharing (same filter everywhere)
  - Translation invariance (detect features anywhere)
  - Hierarchical features (edges → shapes → objects)
""")
```

## Convolution Operation

```python
print("\n=== CONVOLUTION ===")
print("""
CONVOLUTION: Slide a small filter over the image

Filter/Kernel: Small matrix (e.g., 3×3)
Slides across image, computing dot products

Example 3×3 filter:
  [1  0 -1]
  [1  0 -1]
  [1  0 -1]
  
This detects vertical edges!

Output at each position = sum(element-wise × filter × input patch)
""")

def convolve2d(image, kernel):
    """Simple 2D convolution (no padding)"""
    h, w = image.shape
    kh, kw = kernel.shape
    
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(patch * kernel)
    
    return output

# Example: Edge detection
image = np.array([
    [0, 0, 0, 0, 255, 255, 255],
    [0, 0, 0, 0, 255, 255, 255],
    [0, 0, 0, 0, 255, 255, 255],
    [0, 0, 0, 0, 255, 255, 255],
    [0, 0, 0, 0, 255, 255, 255],
], dtype=float)

vertical_edge_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

edge_map = convolve2d(image, vertical_edge_kernel)

print("Input image (5×7):")
print(image.astype(int))
print("\nVertical edge filter (3×3):")
print(vertical_edge_kernel.astype(int))
print("\nEdge detection result:")
print(edge_map.astype(int))
print("\nNote: High values where vertical edge detected!")
```

## Padding and Stride

```python
print("\n=== PADDING AND STRIDE ===")
print("""
PADDING: Add zeros around image border

Why padding?
  - Without: Output smaller than input
  - With 'same' padding: Output same size as input
  
Types:
  - 'valid': No padding (output shrinks)
  - 'same': Pad to keep same dimensions
  
STRIDE: How far filter moves each step

Stride=1: Move one pixel at a time (default)
Stride=2: Move two pixels (downsamples by 2)

Output size formula:
  out_size = (in_size - kernel_size + 2×padding) / stride + 1
""")

def calculate_output_size(input_size, kernel_size, padding, stride):
    """Calculate convolution output size"""
    return (input_size - kernel_size + 2*padding) // stride + 1

print("Example: Input=224, Kernel=3")
print(f"  Padding=0, Stride=1: {calculate_output_size(224, 3, 0, 1)}")
print(f"  Padding=1, Stride=1: {calculate_output_size(224, 3, 1, 1)} (same)")
print(f"  Padding=1, Stride=2: {calculate_output_size(224, 3, 1, 2)} (halved)")
```

## Multiple Filters and Channels

```python
print("\n=== MULTIPLE FILTERS ===")
print("""
Each filter detects a different feature:
  - Filter 1: Vertical edges
  - Filter 2: Horizontal edges
  - Filter 3: Diagonal edges
  - ...

INPUT: H × W × C_in (e.g., 224×224×3 RGB)
FILTERS: K filters, each K_h × K_w × C_in
OUTPUT: H' × W' × K (feature maps)

Example:
  Input: 224 × 224 × 3
  32 filters of size 3 × 3 × 3
  Output: 222 × 222 × 32

Parameters = 32 × (3 × 3 × 3) + 32 biases = 896
  (Much fewer than dense layer!)
""")

# Visualize feature maps concept
print("""
Each filter produces one feature map:

Input Image   →   Filter 1   →   Feature Map 1 (edges)
              →   Filter 2   →   Feature Map 2 (corners)
              →   Filter 3   →   Feature Map 3 (textures)
              →   ...
              →   Filter K   →   Feature Map K
""")
```

## Pooling Layers

```python
print("\n=== POOLING ===")
print("""
POOLING: Reduce spatial dimensions

MAX POOLING:
  - Take maximum value in each window
  - Keeps strongest activations
  - Provides some translation invariance

AVERAGE POOLING:
  - Take average value in each window
  - Smoother downsampling

Typical: 2×2 max pooling with stride 2
  - Reduces dimensions by half
""")

def max_pool_2d(feature_map, pool_size=2, stride=2):
    """2D max pooling"""
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            patch = feature_map[i*stride:i*stride+pool_size,
                               j*stride:j*stride+pool_size]
            output[i, j] = np.max(patch)
    
    return output

feature_map = np.array([
    [1, 2, 5, 6],
    [3, 4, 7, 8],
    [9, 10, 13, 14],
    [11, 12, 15, 16]
], dtype=float)

pooled = max_pool_2d(feature_map)

print("Input feature map (4×4):")
print(feature_map.astype(int))
print("\nAfter 2×2 max pooling (2×2):")
print(pooled.astype(int))
```

## CNN Architecture

```python
print("\n=== CNN ARCHITECTURE ===")
print("""
Typical CNN structure:

INPUT IMAGE
    ↓
[Conv → ReLU → Pool] × N    ← Feature extraction
    ↓
FLATTEN
    ↓
[Dense → ReLU] × M          ← Classification
    ↓
Dense (softmax)
    ↓
OUTPUT CLASSES

Example (simplified VGG-like):

Input: 224×224×3
Conv(64, 3×3) → 224×224×64
MaxPool(2×2) → 112×112×64
Conv(128, 3×3) → 112×112×128
MaxPool(2×2) → 56×56×128
Conv(256, 3×3) → 56×56×256
MaxPool(2×2) → 28×28×256
Flatten → 200,704
Dense(512) → 512
Dense(10) → 10 (output classes)
""")
```

## Building CNN in Keras

```python
print("\n=== CNN IN KERAS ===")
print("""
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # First conv block
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # Second conv block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third conv block
    Conv2D(64, (3, 3), activation='relu'),
    
    # Classification layers
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
""")

# Calculate parameters
print("\nParameter calculation for 32 filters (3×3), input=28×28×1:")
print(f"  Filter weights: 32 × (3 × 3 × 1) = {32 * 3 * 3 * 1}")
print(f"  Biases: 32")
print(f"  Total: {32 * 3 * 3 * 1 + 32}")
```

## Key Points

- **Convolution**: Slide filters to detect local features
- **Parameter sharing**: Same filter applied everywhere
- **Padding**: 'same' keeps dimensions, 'valid' shrinks
- **Stride**: Step size, affects output dimensions
- **Multiple filters**: Each detects different features
- **Pooling**: Reduce dimensions, add invariance
- **Hierarchical**: Layers build complex features from simple ones

## Reflection Questions

1. Why do CNNs need fewer parameters than fully connected networks for images?
2. How does pooling contribute to translation invariance?
3. Why do we stack multiple convolutional layers?
