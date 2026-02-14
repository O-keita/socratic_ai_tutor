# CNN Architectures: VGG, ResNet, and Beyond

## Introduction

Understanding landmark CNN architectures helps you choose the right model for your task and learn principles that apply to designing your own networks.

## VGG Networks

```python
import numpy as np
import pandas as pd

print("=== VGG ARCHITECTURE ===")
print("""
VGG (Visual Geometry Group, 2014):
  - Key insight: Use SMALL 3×3 filters exclusively
  - Stack multiple 3×3 instead of larger filters
  
Why 3×3 everywhere?
  - Two 3×3 = same receptive field as one 5×5
  - But fewer parameters and more non-linearity
  
  5×5: 25 parameters
  3×3 + 3×3: 9 + 9 = 18 parameters (plus extra ReLU!)

VGG16 Structure:
  [Conv3-64] × 2 → MaxPool
  [Conv3-128] × 2 → MaxPool
  [Conv3-256] × 3 → MaxPool
  [Conv3-512] × 3 → MaxPool
  [Conv3-512] × 3 → MaxPool
  Flatten → Dense(4096) → Dense(4096) → Dense(1000)
  
Total: ~138 million parameters
""")

# VGG block
def vgg_block_params(input_channels, output_channels, num_convs):
    """Calculate parameters in a VGG block"""
    total = 0
    in_ch = input_channels
    for _ in range(num_convs):
        total += in_ch * output_channels * 3 * 3 + output_channels
        in_ch = output_channels
    return total

print("VGG16 parameters per block:")
blocks = [
    (3, 64, 2),
    (64, 128, 2),
    (128, 256, 3),
    (256, 512, 3),
    (512, 512, 3),
]

conv_params = 0
for in_ch, out_ch, n in blocks:
    params = vgg_block_params(in_ch, out_ch, n)
    conv_params += params
    print(f"  Block {in_ch}→{out_ch} (×{n}): {params:,}")

fc_params = 7*7*512*4096 + 4096 + 4096*4096 + 4096 + 4096*1000 + 1000
print(f"\nConv layers total: {conv_params:,}")
print(f"FC layers total: {fc_params:,}")
print(f"Most params in FC layers!")
```

## GoogLeNet / Inception

```python
print("\n=== INCEPTION MODULE ===")
print("""
GoogLeNet/Inception (2014):
  - Use multiple filter sizes in parallel
  - Let the network learn which scales matter
  
Inception module:
         ┌→ 1×1 Conv ──────────────────────────┐
  Input ─┼→ 1×1 Conv → 3×3 Conv ───────────────┼→ Concat
         ├→ 1×1 Conv → 5×5 Conv ───────────────┤
         └→ 3×3 MaxPool → 1×1 Conv ────────────┘

Key ideas:
  1. MULTI-SCALE: 1×1, 3×3, 5×5 in parallel
  2. BOTTLENECK: 1×1 conv reduces channels before expensive convs
  3. EFFICIENCY: Only 5M parameters (vs 138M in VGG)
""")

def inception_module_params(input_channels, branches):
    """Calculate Inception module parameters"""
    total = 0
    # branches: list of (output_channels, [(kernel, output)...])
    for branch in branches:
        in_ch = input_channels
        for kernel, out_ch in branch:
            total += in_ch * out_ch * kernel * kernel + out_ch
            in_ch = out_ch
    return total

# Example inception module
branches = [
    [(1, 64)],           # 1×1
    [(1, 96), (3, 128)], # 1×1 → 3×3
    [(1, 16), (5, 32)],  # 1×1 → 5×5
    [(3, 32)]            # pool → 1×1 (simplified)
]

params = inception_module_params(256, branches)
print(f"\nInception module (256 input): {params:,} params")
```

## ResNet: Residual Connections

```python
print("\n=== RESNET ===")
print("""
PROBLEM: Very deep networks are hard to train
  - Vanishing gradients
  - Degradation: deeper networks have HIGHER training error

SOLUTION: Residual/Skip connections

Standard block:      Residual block:
  x                    x ─────────┐
  ↓                    ↓          │
  Conv                 Conv       │
  ↓                    ↓          │
  Conv                 Conv       │
  ↓                    ↓          │
  output               + ←────────┘
                       ↓
                       output = F(x) + x

Now the network learns: F(x) = desired_output - x
  - If identity is optimal, F(x) → 0 (easy to learn!)
  - Gradients flow directly through skip connection

Enables training of 100+ layers!
""")

def resnet_block(input_channels, output_channels, stride=1):
    """Describe ResNet basic block"""
    params = 0
    
    # If dimensions change, need projection shortcut
    shortcut_params = 0
    if stride > 1 or input_channels != output_channels:
        shortcut_params = input_channels * output_channels + output_channels
    
    # Two 3×3 convolutions
    params += input_channels * output_channels * 9 + output_channels
    params += output_channels * output_channels * 9 + output_channels
    
    return params + shortcut_params

print("ResNet-34 structure:")
print("""
  Input: 224×224×3
  Conv(64, 7×7, stride=2) → 112×112×64
  MaxPool(3×3, stride=2) → 56×56×64
  
  Layer1: [Basic(64)] × 3  → 56×56×64
  Layer2: [Basic(128)] × 4 → 28×28×128
  Layer3: [Basic(256)] × 6 → 14×14×256
  Layer4: [Basic(512)] × 3 → 7×7×512
  
  GlobalAvgPool → 512
  Dense(1000) → 1000 classes
""")
```

## Bottleneck Block

```python
print("\n=== BOTTLENECK BLOCK ===")
print("""
For deeper ResNets (50+), use bottleneck:

Basic block:           Bottleneck block:
  3×3, 64                1×1, 64  (reduce)
  3×3, 64                3×3, 64  (conv)
                         1×1, 256 (expand)

Bottleneck is MORE efficient despite 3 layers:
  Basic: 64×64×9 + 64×64×9 = 73,728
  Bottleneck: 256×64 + 64×64×9 + 64×256 = 69,632

Standard configurations:
  ResNet-18/34: Basic blocks
  ResNet-50/101/152: Bottleneck blocks
""")

print("\nResNet comparison:")
configs = {
    'ResNet-18': ('Basic', [2, 2, 2, 2]),
    'ResNet-34': ('Basic', [3, 4, 6, 3]),
    'ResNet-50': ('Bottleneck', [3, 4, 6, 3]),
    'ResNet-101': ('Bottleneck', [3, 4, 23, 3]),
    'ResNet-152': ('Bottleneck', [3, 8, 36, 3]),
}

for name, (block_type, layers) in configs.items():
    total_blocks = sum(layers)
    print(f"  {name}: {block_type} × {layers} = {total_blocks} blocks")
```

## Modern Architectures

```python
print("\n=== MODERN ARCHITECTURES ===")
print("""
DenseNet (2017):
  - Connect each layer to ALL previous layers
  - Encourages feature reuse
  - Very efficient parameters
  
EfficientNet (2019):
  - Compound scaling: depth, width, resolution together
  - Neural Architecture Search (NAS) optimized
  - Best accuracy/params trade-off
  
ConvNeXt (2022):
  - "Modernized" ResNet with Transformer ideas
  - Larger kernels (7×7), LayerNorm, GELU
  - Competitive with Vision Transformers

Vision Transformer (ViT, 2020):
  - Attention-based, no convolutions
  - Patch embedding + Transformer
  - Dominates with large data/compute
""")

# Architecture comparison
print("\nImageNet comparison:")
print(f"{'Model':<15} {'Top-1 Acc':<12} {'Params':<15}")
print("-" * 42)
models = [
    ('VGG-16', '71.5%', '138M'),
    ('ResNet-50', '76.1%', '26M'),
    ('ResNet-152', '77.8%', '60M'),
    ('EfficientNet-B0', '77.3%', '5.3M'),
    ('EfficientNet-B7', '84.4%', '66M'),
    ('ViT-L/16', '85.2%', '307M'),
]
for name, acc, params in models:
    print(f"{name:<15} {acc:<12} {params:<15}")
```

## Transfer Learning

```python
print("\n=== TRANSFER LEARNING ===")
print("""
Pre-trained CNN as feature extractor:

1. FEATURE EXTRACTION:
   - Load pre-trained model (e.g., ResNet on ImageNet)
   - Remove final classification layer
   - Add new classifier for your task
   - Freeze pre-trained weights, train only new layers
   
2. FINE-TUNING:
   - Start with feature extraction
   - Unfreeze some/all pre-trained layers
   - Train with small learning rate

Code example (Keras):

base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
""")
```

## Choosing an Architecture

```python
print("\n=== CHOOSING AN ARCHITECTURE ===")
print("""
Guidelines:

SMALL DATASET (<1000 samples):
  → Transfer learning from ImageNet
  → Don't train from scratch

MEDIUM DATASET (1K-100K):
  → EfficientNet-B0 to B3
  → Fine-tuned ResNet-50

LARGE DATASET (100K+):
  → ResNet-50/101
  → EfficientNet
  → Consider ViT if very large

MOBILE/EDGE:
  → MobileNet
  → EfficientNet-Lite
  → Quantized models

ACCURACY CRITICAL:
  → Ensemble multiple models
  → EfficientNet-B7 or larger
  → ViT with extra data

SPEED CRITICAL:
  → MobileNet
  → EfficientNet-B0
  → ResNet-18
""")
```

## Key Points

- **VGG**: Simple, uniform 3×3 convs, many parameters
- **Inception**: Multi-scale parallel paths, efficient
- **ResNet**: Skip connections enable very deep networks
- **Bottleneck**: 1×1 reduce → 3×3 → 1×1 expand, efficient
- **EfficientNet**: Compound scaling, best accuracy/params
- **Transfer learning**: Use pre-trained models, fine-tune

## Reflection Questions

1. Why do residual connections enable training of deeper networks?
2. How does the bottleneck design reduce computation while maintaining performance?
3. When should you use transfer learning vs training from scratch?
