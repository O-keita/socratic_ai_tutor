# Self-Supervised Learning

## Introduction

Self-supervised learning creates training signals from the data itself, without human labels. This enables learning from massive amounts of unlabeled data, which is abundant and cheap.

## Why Self-Supervised Learning?

```python
import numpy as np
import pandas as pd

print("=== THE LABEL BOTTLENECK ===")
print("""
SUPERVISED LEARNING NEEDS LABELS:
  - ImageNet: 14M images, 22K categories
  - Years of human annotation effort
  - Expensive: $0.01-$10 per label
  - Noisy: Human annotators disagree

UNLABELED DATA IS ABUNDANT:
  - YouTube: 500 hours uploaded per minute
  - Internet: Billions of images
  - Text: Entire web's content
  
SELF-SUPERVISED LEARNING:
  - Create supervision from data itself
  - No human labels needed
  - Scale to massive datasets
  - Learn general representations

SUCCESS STORIES:
  - GPT: Predict next word
  - BERT: Fill in masked words
  - SimCLR/CLIP: Contrastive learning
  - MAE: Masked autoencoders
""")
```

## Pretext Tasks

```python
print("\n=== PRETEXT TASKS ===")
print("""
Pretext task: Artificial task that creates supervision

IMAGES:
1. Rotation prediction
   - Rotate image 0°, 90°, 180°, 270°
   - Predict rotation angle
   
2. Jigsaw puzzles
   - Split image into patches
   - Shuffle and predict arrangement
   
3. Colorization
   - Convert to grayscale
   - Predict original colors
   
4. Inpainting
   - Mask part of image
   - Predict missing region

5. Context prediction
   - Sample two patches
   - Predict relative position

TEXT:
1. Masked language modeling (BERT)
   - Mask random words
   - Predict masked words
   
2. Next sentence prediction
   - Given two sentences
   - Predict if second follows first
   
3. Next word prediction (GPT)
   - Given prefix
   - Predict next word

AUDIO:
1. Predict future frames
2. Audio-visual correspondence
""")

# Simple rotation pretext task demo
def create_rotation_task(images):
    """Create rotation prediction dataset"""
    X = []
    y = []
    
    for img in images:
        for rotation, label in [(0, 0), (1, 1), (2, 2), (3, 3)]:
            rotated = np.rot90(img, k=rotation)
            X.append(rotated)
            y.append(label)
    
    return np.array(X), np.array(y)

# Demo with random "images"
np.random.seed(42)
fake_images = np.random.randn(5, 8, 8)  # 5 images, 8x8

X_rot, y_rot = create_rotation_task(fake_images)
print(f"Rotation task: {len(X_rot)} samples, {len(np.unique(y_rot))} classes")
```

## Contrastive Learning

```python
print("\n=== CONTRASTIVE LEARNING ===")
print("""
Key idea: Learn by comparing similar and dissimilar examples

POSITIVE PAIRS: Different views of same image
  - Random crop
  - Color jitter
  - Flip, rotation
  → Should have similar representations

NEGATIVE PAIRS: Different images
  → Should have different representations

LOSS FUNCTION (InfoNCE):
  L = -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))
  
  Where:
  - z_i, z_j: Positive pair (same image, different augmentations)
  - z_k: Negatives (other images)
  - τ: Temperature (controls sharpness)
  - sim(): Cosine similarity

SIMCLR FRAMEWORK:
  1. Augment image twice → x_i, x_j
  2. Encode: f(x_i) → h_i, f(x_j) → h_j
  3. Project: g(h_i) → z_i, g(h_j) → z_j
  4. Contrastive loss on z_i, z_j
""")

def simclr_loss(z_i, z_j, temperature=0.5):
    """Simplified SimCLR loss for a batch"""
    batch_size = len(z_i)
    
    # Normalize
    z_i = z_i / np.linalg.norm(z_i, axis=1, keepdims=True)
    z_j = z_j / np.linalg.norm(z_j, axis=1, keepdims=True)
    
    # Concatenate all representations
    z = np.vstack([z_i, z_j])  # 2N x D
    
    # Compute similarity matrix
    sim_matrix = z @ z.T / temperature  # 2N x 2N
    
    # Create labels (positive pairs)
    labels = np.arange(batch_size)
    labels = np.concatenate([labels + batch_size, labels])  # [N:2N, 0:N]
    
    # Mask self-similarity
    mask = np.eye(2 * batch_size, dtype=bool)
    sim_matrix[mask] = -np.inf
    
    # Cross-entropy loss
    exp_sim = np.exp(sim_matrix)
    loss = -np.log(exp_sim[np.arange(2*batch_size), labels] / exp_sim.sum(axis=1))
    
    return loss.mean()

# Demo
batch_size = 4
dim = 128

z_i = np.random.randn(batch_size, dim)
z_j = z_i + np.random.randn(batch_size, dim) * 0.1  # Similar views

loss = simclr_loss(z_i, z_j, temperature=0.5)
print(f"SimCLR loss: {loss:.4f}")
```

## CLIP: Connecting Vision and Language

```python
print("\n=== CLIP ===")
print("""
Contrastive Language-Image Pre-training

IDEA: Learn from image-text pairs
  - Web has billions of image-caption pairs
  - Learn joint embedding space

TRAINING:
  Images: I_1, I_2, ..., I_N
  Texts:  T_1, T_2, ..., T_N
  
  Positive pairs: (I_i, T_i) should be similar
  Negative pairs: (I_i, T_j) where i≠j should be dissimilar

ARCHITECTURE:
  Image Encoder (ViT or ResNet) → Image embedding
  Text Encoder (Transformer) → Text embedding
  
  Contrastive loss on embeddings

ZERO-SHOT CLASSIFICATION:
  1. Encode class names: "a photo of a dog", "a photo of a cat"
  2. Encode image
  3. Find most similar text embedding
  
  No training on specific classes needed!

REMARKABLE RESULTS:
  - Competitive with supervised on ImageNet
  - Generalizes to many tasks
  - Robust to distribution shift
""")

def clip_zero_shot(image_embedding, class_embeddings, class_names):
    """CLIP-style zero-shot classification"""
    # Normalize
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    class_embeddings = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = class_embeddings @ image_embedding
    
    # Return class with highest similarity
    best_idx = np.argmax(similarities)
    return class_names[best_idx], similarities[best_idx]

# Demo
np.random.seed(42)
image_emb = np.random.randn(512)
class_embs = np.random.randn(5, 512)
class_names = ['dog', 'cat', 'car', 'airplane', 'bird']

pred_class, score = clip_zero_shot(image_emb, class_embs, class_names)
print(f"Zero-shot prediction: {pred_class} (score: {score:.3f})")
```

## Masked Autoencoders (MAE)

```python
print("\n=== MASKED AUTOENCODERS ===")
print("""
Key idea: Mask most of the image, reconstruct it

PROCESS:
1. Split image into patches (e.g., 16x16)
2. Randomly mask 75% of patches
3. Encode visible patches with ViT
4. Decode: reconstruct masked patches
5. Loss: MSE on masked patch pixels

WHY HIGH MASKING RATIO?
  - Forces learning of semantics
  - Can't just interpolate from neighbors
  - More efficient training

ARCHITECTURE:
  - Asymmetric encoder-decoder
  - Encoder: Only on visible patches (small)
  - Decoder: On full image with mask tokens (can be small)

EFFICIENCY:
  - 3x+ faster than contrastive methods
  - Encoder processes only 25% of patches

RESULTS:
  - State-of-the-art on ImageNet
  - Excellent transfer learning
  - Works for video too (VideoMAE)
""")

def mae_mask_patches(image_patches, mask_ratio=0.75):
    """Mask patches for MAE"""
    n_patches = len(image_patches)
    n_mask = int(n_patches * mask_ratio)
    
    # Random mask indices
    mask_indices = np.random.choice(n_patches, n_mask, replace=False)
    visible_indices = np.setdiff1d(np.arange(n_patches), mask_indices)
    
    visible_patches = image_patches[visible_indices]
    
    return visible_patches, visible_indices, mask_indices

# Demo
np.random.seed(42)
# Simulate 196 patches (14x14 grid) from 224x224 image with 16x16 patches
n_patches = 196
patch_dim = 256  # Flattened 16x16
patches = np.random.randn(n_patches, patch_dim)

visible, vis_idx, mask_idx = mae_mask_patches(patches, mask_ratio=0.75)

print(f"Total patches: {n_patches}")
print(f"Visible patches: {len(visible)} ({len(visible)/n_patches*100:.0f}%)")
print(f"Masked patches: {len(mask_idx)} ({len(mask_idx)/n_patches*100:.0f}%)")
```

## Self-Supervised Learning Pipeline

```python
print("\n=== SSL PIPELINE ===")
print("""
TYPICAL WORKFLOW:

1. PRE-TRAINING (Self-supervised):
   - Large unlabeled dataset
   - Pretext task or contrastive learning
   - Learn general representations
   - Long training (days/weeks)

2. FINE-TUNING (Supervised):
   - Small labeled dataset
   - Standard supervised learning
   - Adapt to specific task
   - Quick training (hours)

OPTIONS FOR FINE-TUNING:

a) LINEAR PROBE:
   - Freeze encoder
   - Train linear classifier on top
   - Tests representation quality

b) FULL FINE-TUNING:
   - Unfreeze all layers
   - Fine-tune everything
   - Best task performance

c) PARTIAL FINE-TUNING:
   - Freeze early layers
   - Fine-tune later layers
   - Balance between probe and full

EVALUATION PROTOCOL:
1. Pre-train on ImageNet (no labels)
2. Linear probe: 76% → Good representations
3. Fine-tune: 85% → Great for downstream
4. Transfer to other datasets
""")

print("""
COMPARISON OF METHODS:

Method          | Pretext Task              | Key Innovation
----------------|---------------------------|------------------
SimCLR          | Contrastive               | Strong augmentations
MoCo            | Contrastive               | Momentum encoder
BYOL            | Prediction                | No negatives needed
SwAV            | Clustering                | Online clustering
DINO            | Self-distillation         | Attention emerges
MAE             | Reconstruction            | High masking ratio
CLIP            | Image-text matching       | Web-scale pairs
""")
```

## Key Points

- **Self-supervised**: Create labels from data itself
- **Pretext tasks**: Rotation, jigsaw, masking, etc.
- **Contrastive**: Pull similar together, push different apart
- **CLIP**: Connect images and text for zero-shot
- **MAE**: Mask and reconstruct for efficient learning
- **Pipeline**: Pre-train → Fine-tune

## Reflection Questions

1. Why do self-supervised methods often outperform supervised when data is limited?
2. What makes a good pretext task for learning useful representations?
3. How does self-supervised learning change what data we collect?
