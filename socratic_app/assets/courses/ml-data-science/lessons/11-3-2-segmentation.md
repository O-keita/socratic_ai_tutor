# Semantic Segmentation

## Introduction

Semantic segmentation classifies every pixel in an image, providing dense predictions. Unlike detection, it produces pixel-level understanding of scenes.

## Segmentation Types

```python
import numpy as np
import pandas as pd

print("=== SEGMENTATION TYPES ===")
print("""
SEMANTIC SEGMENTATION:
  - Label each pixel with a class
  - All 'person' pixels same class
  - No distinction between instances
  
INSTANCE SEGMENTATION:
  - Separate each object instance
  - Person 1 vs Person 2
  - Combines detection + segmentation
  
PANOPTIC SEGMENTATION:
  - Unified: stuff + things
  - Stuff: sky, road (semantic)
  - Things: cars, people (instance)

Output shapes:
  Input: H × W × 3
  Semantic: H × W × num_classes (or H × W with class IDs)
  Instance: H × W with instance IDs + masks
""")
```

## Fully Convolutional Networks (FCN)

```python
print("\n=== FULLY CONVOLUTIONAL NETWORKS ===")
print("""
FCN (2015): First deep learning approach

Key insight: Replace FC layers with convolutions
  - Dense(4096) → Conv(4096, 1×1)
  - Allows any input size
  - Output is spatial map

Architecture:
  1. Encoder: VGG-like, reduces resolution
  2. Decoder: Upsampling to full resolution

Input: H × W × 3
Encoder: H/32 × W/32 × 512
Decoder: H × W × num_classes

Problem: Lost fine details during downsampling
Solution: Skip connections from encoder
""")

print("""
FCN variants:

FCN-32s: Single 32× upsample (coarse)
FCN-16s: Skip from pool4, 16× upsample (better)
FCN-8s: Skips from pool3, pool4, 8× upsample (best)

Skip connections preserve:
  - Edge information
  - Fine spatial details
""")
```

## Encoder-Decoder Architecture

```python
print("\n=== ENCODER-DECODER ===")
print("""
General structure:

ENCODER (Contracting path):
  - Conv + Pool → reduce spatial dims
  - Increase channels
  - Extract features at multiple scales

DECODER (Expanding path):
  - Upsample + Conv → increase spatial dims
  - Decrease channels
  - Recover spatial resolution

Popular architectures:
  - SegNet: Max-pooling indices for upsampling
  - U-Net: Skip connections between encoder-decoder
  - DeepLab: Dilated convolutions, no pooling
""")
```

## U-Net Architecture

```python
print("\n=== U-NET ===")
print("""
U-Net (2015): Designed for biomedical images

U-shaped architecture:
  - Symmetric encoder-decoder
  - Skip connections at each level
  - Concatenate encoder features with decoder

       Encoder             Decoder
    ┌─────────┐           ┌─────────┐
    │ Conv×2  │─────────→ │ Conv×2  │
    └────┬────┘           └────┬────┘
         ↓ Pool                ↑ Up
    ┌─────────┐           ┌─────────┐
    │ Conv×2  │─────────→ │ Conv×2  │
    └────┬────┘           └────┬────┘
         ↓ Pool                ↑ Up
    ┌─────────┐           ┌─────────┐
    │ Conv×2  │─────────→ │ Conv×2  │
    └────┬────┘           └────┬────┘
         ↓ Pool                ↑ Up
    ┌─────────┐           ┌─────────┐
    │ Conv×2  │─────────→ │ Conv×2  │
    └────┬────┘           └────┬────┘
         └──→ Bottleneck ──→──┘

Skip connections preserve fine details!
""")

print("""
U-Net in Keras:

def unet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    
    # Bottleneck
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    
    # Decoder
    u1 = UpSampling2D()(c3)
    u1 = concatenate([u1, c2])  # Skip connection!
    c4 = Conv2D(128, 3, activation='relu', padding='same')(u1)
    
    u2 = UpSampling2D()(c4)
    u2 = concatenate([u2, c1])  # Skip connection!
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u2)
    
    outputs = Conv2D(num_classes, 1, activation='softmax')(c5)
    
    return Model(inputs, outputs)
""")
```

## Dilated (Atrous) Convolutions

```python
print("\n=== DILATED CONVOLUTIONS FOR SEGMENTATION ===")
print("""
Problem with pooling:
  - Reduces resolution
  - Loses fine details
  - Need to upsample

Alternative: Dilated convolutions
  - No resolution loss
  - Larger receptive field
  - Same number of parameters

DeepLab uses Atrous Spatial Pyramid Pooling (ASPP):
  - Multiple dilated convs with different rates
  - Captures multi-scale context
  - rate=1, 6, 12, 18 in parallel
""")

def receptive_field_with_dilation(layers):
    """Calculate receptive field with dilated convs"""
    rf = 1
    for kernel, dilation in layers:
        effective_kernel = kernel + (kernel - 1) * (dilation - 1)
        rf += effective_kernel - 1
    return rf

# Regular convs
regular = [(3, 1), (3, 1), (3, 1), (3, 1)]  # rate=1
rf_regular = receptive_field_with_dilation(regular)

# Dilated convs
dilated = [(3, 1), (3, 2), (3, 4), (3, 8)]  # increasing rates
rf_dilated = receptive_field_with_dilation(dilated)

print(f"4 × 3×3 conv (no dilation): RF = {rf_regular}")
print(f"4 × 3×3 conv (with dilation): RF = {rf_dilated}")
print("Same parameters, larger context!")
```

## Loss Functions for Segmentation

```python
print("\n=== LOSS FUNCTIONS ===")
print("""
CROSS-ENTROPY (per-pixel):
  Standard classification loss at each pixel
  L = -Σ y_true × log(y_pred)  for each pixel

DICE LOSS:
  Based on Dice coefficient (overlap measure)
  Dice = 2 × |A ∩ B| / (|A| + |B|)
  Loss = 1 - Dice
  Good for imbalanced classes

FOCAL LOSS:
  Down-weights easy examples
  Focuses on hard pixels
  FL = -α(1-p)^γ × log(p)

COMBINED:
  Total loss = CE + λ × Dice
  Often works better than either alone
""")

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient"""
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Example
y_true = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
y_pred = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])

dice = dice_coefficient(y_true, y_pred)
print(f"Ground truth:\n{y_true}")
print(f"\nPrediction:\n{y_pred}")
print(f"\nDice coefficient: {dice:.3f}")
print(f"Dice loss: {1-dice:.3f}")
```

## Evaluation Metrics

```python
print("\n=== EVALUATION METRICS ===")
print("""
PIXEL ACCURACY:
  Correct pixels / Total pixels
  Simple but misleading with class imbalance

MEAN IoU (mIoU):
  Average IoU across all classes
  IoU = TP / (TP + FP + FN)
  Standard metric for segmentation

DICE/F1 SCORE:
  Dice = 2×TP / (2×TP + FP + FN)
  Common in medical imaging

Per-class metrics important for:
  - Rare classes
  - Critical classes (medical)
""")

def calculate_iou(pred, target, num_classes):
    """Calculate IoU per class"""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = np.sum(pred_c & target_c)
        union = np.sum(pred_c | target_c)
        
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))
    
    return ious

# Example
pred = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 1]])
target = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 2]])

ious = calculate_iou(pred, target, 3)
print(f"Per-class IoU: {[round(x, 3) if not np.isnan(x) else 'N/A' for x in ious]}")
print(f"mIoU: {np.nanmean(ious):.3f}")
```

## Key Points

- **Semantic segmentation**: Per-pixel classification
- **FCN**: Fully convolutional, skip connections for details
- **U-Net**: Symmetric encoder-decoder, concatenation skips
- **Dilated convolutions**: Large receptive field without pooling
- **Dice loss**: Better for imbalanced classes
- **mIoU**: Standard evaluation metric

## Reflection Questions

1. Why are skip connections crucial for segmentation accuracy?
2. How do dilated convolutions help preserve spatial resolution?
3. When would you prefer Dice loss over cross-entropy?
