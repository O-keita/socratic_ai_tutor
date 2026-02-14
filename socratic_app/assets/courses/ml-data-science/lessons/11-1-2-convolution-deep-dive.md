# Convolutional Layers Deep Dive

## Introduction

Convolutional layers are the building blocks of CNNs. Understanding their parameters, behavior, and variations is essential for designing effective image models.

## Convolution Parameters

```python
import numpy as np
import pandas as pd

print("=== CONVOLUTION PARAMETERS ===")
print("""
Key parameters of Conv2D:

1. FILTERS (out_channels):
   Number of different features to detect
   More filters = more features = more parameters

2. KERNEL_SIZE:
   Size of sliding window (e.g., 3×3, 5×5)
   Larger = sees more context, more parameters

3. STRIDE:
   Step size when sliding filter
   stride=1: Dense sliding
   stride=2: Downsamples by 2

4. PADDING:
   'valid': No padding (output shrinks)
   'same': Pad to keep same spatial size

5. ACTIVATION:
   'relu' most common
   Can be None (add separately)
""")

def conv_output_shape(input_shape, filters, kernel_size, stride, padding):
    """Calculate output shape of Conv2D"""
    H, W, C = input_shape
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    
    if padding == 'same':
        pad = (kH - 1) // 2
    else:  # 'valid'
        pad = 0
    
    out_H = (H - kH + 2*pad) // stride + 1
    out_W = (W - kW + 2*pad) // stride + 1
    
    return (out_H, out_W, filters)

# Examples
print("\nExample output shapes:")
input_shape = (224, 224, 3)

configs = [
    (64, 3, 1, 'same'),
    (64, 3, 1, 'valid'),
    (64, 3, 2, 'same'),
    (128, 5, 1, 'same'),
]

for filters, ks, stride, pad in configs:
    out = conv_output_shape(input_shape, filters, ks, stride, pad)
    print(f"  Conv({filters}, {ks}×{ks}, stride={stride}, {pad}): {input_shape} → {out}")
```

## Parameter Counting

```python
print("\n=== PARAMETER COUNTING ===")
print("""
Conv2D parameters:

Weights: filters × kernel_H × kernel_W × input_channels
Biases: filters

Total = filters × (kH × kW × C_in + 1)

Example: Conv2D(64, (3,3)) with 3 input channels
  Weights: 64 × 3 × 3 × 3 = 1,728
  Biases: 64
  Total: 1,792 parameters
""")

def count_conv_params(input_channels, filters, kernel_size, use_bias=True):
    """Count parameters in Conv2D layer"""
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    weights = filters * kH * kW * input_channels
    biases = filters if use_bias else 0
    return weights + biases

print("Parameter counts:")
print(f"  Conv(32, 3×3), input=3 channels: {count_conv_params(3, 32, 3):,}")
print(f"  Conv(64, 3×3), input=32 channels: {count_conv_params(32, 64, 3):,}")
print(f"  Conv(128, 3×3), input=64 channels: {count_conv_params(64, 128, 3):,}")
print(f"  Conv(256, 3×3), input=128 channels: {count_conv_params(128, 256, 3):,}")

print("\nCompare to Dense layer:")
# Flatten 7×7×512 to Dense(4096)
flatten_size = 7 * 7 * 512
dense_params = flatten_size * 4096 + 4096
print(f"  Dense(4096), input=7×7×512: {dense_params:,} parameters!")
```

## 1×1 Convolutions

```python
print("\n=== 1×1 CONVOLUTIONS ===")
print("""
1×1 convolution: Kernel size of 1

What it does:
  - No spatial mixing (just one pixel)
  - Mixes channels at each position
  - Like a Dense layer applied to each pixel

Uses:
  1. REDUCE CHANNELS (compression):
     Input: H×W×256 → 1×1 Conv(64) → H×W×64
     
  2. INCREASE NON-LINEARITY:
     Add another activation layer cheaply
     
  3. BOTTLENECK:
     Used in ResNet, Inception
     Reduce dims before expensive 3×3 conv
""")

# Channel reduction example
print("Channel reduction example:")
print(f"  Input: 56×56×256")
print(f"  1×1 Conv(64): {count_conv_params(256, 64, 1):,} params")
print(f"  Output: 56×56×64")
print(f"\n  vs 3×3 Conv(64): {count_conv_params(256, 64, 3):,} params")
```

## Depthwise Separable Convolutions

```python
print("\n=== DEPTHWISE SEPARABLE CONVOLUTION ===")
print("""
Split regular convolution into two steps:

REGULAR CONV:
  Input: H×W×C_in
  Filter: K×K×C_in×C_out
  Params: K×K×C_in×C_out

DEPTHWISE SEPARABLE:
  1. Depthwise: One K×K filter PER input channel
     Params: K×K×C_in
  
  2. Pointwise: 1×1 conv to mix channels
     Params: C_in×C_out

Total: K×K×C_in + C_in×C_out

SAVINGS:
  Regular: K²×C_in×C_out
  Separable: K²×C_in + C_in×C_out
  Ratio ≈ 1/C_out + 1/K²
  
For K=3, C_out=256: ~8-9x fewer parameters!
""")

def separable_params(input_channels, output_channels, kernel_size):
    """Parameters for depthwise separable conv"""
    K = kernel_size
    depthwise = K * K * input_channels
    pointwise = input_channels * output_channels
    return depthwise + pointwise

print("Parameter comparison:")
C_in, C_out, K = 64, 128, 3
regular = count_conv_params(C_in, C_out, K)
separable = separable_params(C_in, C_out, K)

print(f"  Regular 3×3 Conv: {regular:,} params")
print(f"  Separable Conv: {separable:,} params")
print(f"  Savings: {regular / separable:.1f}x fewer")
```

## Dilated (Atrous) Convolutions

```python
print("\n=== DILATED CONVOLUTIONS ===")
print("""
Insert gaps in the kernel:

Regular 3×3:    Dilated 3×3 (rate=2):
  [* * *]           [*   *   *]
  [* * *]           [         ]
  [* * *]           [*   *   *]
                    [         ]
                    [*   *   *]

Dilation rate: Spacing between kernel elements

Benefits:
  - Larger receptive field WITHOUT more params
  - 3×3 with rate=2 covers 5×5 area
  - Good for dense prediction (segmentation)

Effective kernel size = K + (K-1)×(rate-1)
  3×3 rate=1: 3
  3×3 rate=2: 5
  3×3 rate=4: 9
""")

def effective_kernel_size(kernel_size, dilation_rate):
    """Calculate effective kernel size with dilation"""
    return kernel_size + (kernel_size - 1) * (dilation_rate - 1)

print("Effective receptive field:")
for rate in [1, 2, 4, 8]:
    eff = effective_kernel_size(3, rate)
    print(f"  3×3 with dilation={rate}: covers {eff}×{eff} area")
```

## Transposed Convolutions

```python
print("\n=== TRANSPOSED CONVOLUTION ===")
print("""
UPSAMPLING with learnable filters:

Regular conv: H×W → smaller
Transposed conv: H×W → larger (upsamples)

Also called:
  - Deconvolution (technically incorrect)
  - Fractionally-strided convolution

Uses:
  - Image generation (GANs, autoencoders)
  - Semantic segmentation (decode to full resolution)
  
Conv2DTranspose(filters, kernel_size, strides=2):
  Input: H×W → Output: 2H×2W (doubles dimensions)
""")

def transposed_output_shape(input_shape, filters, kernel_size, stride, padding='same'):
    """Calculate transposed conv output shape"""
    H, W, C = input_shape
    
    if padding == 'same':
        out_H = H * stride
        out_W = W * stride
    else:  # 'valid'
        out_H = H * stride + max(kernel_size - stride, 0)
        out_W = W * stride + max(kernel_size - stride, 0)
    
    return (out_H, out_W, filters)

print("Transposed conv examples (upsampling):")
input_shape = (7, 7, 512)
out = transposed_output_shape(input_shape, 256, 4, 2)
print(f"  {input_shape} → ConvTranspose(256, 4×4, stride=2) → {out}")

input_shape = (14, 14, 256)
out = transposed_output_shape(input_shape, 128, 4, 2)
print(f"  {input_shape} → ConvTranspose(128, 4×4, stride=2) → {out}")
```

## Key Points

- **Kernel size**: 3×3 most common (balance of receptive field and params)
- **Stride**: Use for downsampling instead of pooling in modern architectures
- **1×1 conv**: Mix channels, reduce dimensions cheaply
- **Depthwise separable**: Dramatic parameter reduction, same performance
- **Dilated**: Increase receptive field without more parameters
- **Transposed**: Learnable upsampling for generation/segmentation

## Reflection Questions

1. Why is 3×3 the most popular kernel size in modern CNNs?
2. When would you use depthwise separable convolutions vs regular convolutions?
3. How can dilated convolutions help with dense prediction tasks?
