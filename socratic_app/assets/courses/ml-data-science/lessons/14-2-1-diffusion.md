# Diffusion Models

## Introduction

Diffusion models generate data by learning to reverse a gradual noising process. They have achieved state-of-the-art results in image generation, powering tools like DALL-E, Stable Diffusion, and Midjourney.

## The Diffusion Process

```python
import numpy as np
import pandas as pd

print("=== DIFFUSION INTUITION ===")
print("""
TWO PROCESSES:

1. FORWARD PROCESS (Fixed, no learning):
   - Start with data x₀
   - Gradually add Gaussian noise
   - After T steps: pure noise x_T ~ N(0, I)
   
   x₀ → x₁ → x₂ → ... → x_T
   (clean)              (noise)

2. REVERSE PROCESS (Learned):
   - Start with pure noise x_T
   - Gradually denoise
   - Recover clean data x₀
   
   x_T → x_{T-1} → ... → x₀
   (noise)           (clean)

Training: Learn to reverse the noise addition
Generation: Sample noise, then denoise
""")
```

## Forward Diffusion

```python
print("\n=== FORWARD PROCESS ===")
print("""
Forward process adds noise step by step:

q(x_t | x_{t-1}) = N(x_t; √(1-β_t) × x_{t-1}, β_t × I)

β_t: Noise schedule (variance at step t)
  - Small at start (~0.0001)
  - Larger later (~0.02)

Key property - jump to any step directly:
q(x_t | x_0) = N(x_t; √ᾱ_t × x_0, (1-ᾱ_t) × I)

where ᾱ_t = ∏_{s=1}^t (1 - β_s)

This allows efficient training!
""")

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Sample x_t given x_0 directly"""
    noise = np.random.randn(*x_0.shape)
    
    sqrt_alpha_bar = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]
    
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    return x_t, noise

# Setup noise schedule
T = 1000
beta = np.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

sqrt_alpha_bar = np.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)

# Example
x_0 = np.array([0.5, 0.3, -0.2])

print("Noise levels at different timesteps:")
for t in [0, 100, 500, 999]:
    x_t, _ = forward_diffusion_sample(x_0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
    signal_ratio = sqrt_alpha_bar[t]
    print(f"  t={t:4d}: signal_ratio={signal_ratio:.4f}, x_t={x_t.round(3)}")
```

## Reverse Process

```python
print("\n=== REVERSE PROCESS ===")
print("""
Learning to denoise:

p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

Neural network predicts:
  - Original: μ_θ directly
  - DDPM approach: ε_θ (the noise)
  
Given noise prediction ε_θ:
  μ_θ = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ(x_t, t))

Training objective (simplified):
  L = E[||ε - ε_θ(x_t, t)||²]
  
  Sample t uniformly
  Add noise to get x_t
  Predict the noise
  MSE loss
""")

def reverse_step(x_t, t, predicted_noise, alpha, alpha_bar, beta):
    """One reverse diffusion step"""
    sqrt_alpha = np.sqrt(alpha[t])
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])
    
    # Mean
    mean = (1 / sqrt_alpha) * (x_t - (beta[t] / sqrt_one_minus_alpha_bar) * predicted_noise)
    
    # Add noise (except for t=0)
    if t > 0:
        noise = np.random.randn(*x_t.shape)
        sigma = np.sqrt(beta[t])
        x_prev = mean + sigma * noise
    else:
        x_prev = mean
    
    return x_prev

print("Reverse process: Start from noise, denoise step by step")
print("  x_T (pure noise) → x_{T-1} → ... → x_1 → x_0 (clean)")
```

## U-Net Architecture

```python
print("\n=== DIFFUSION MODEL ARCHITECTURE ===")
print("""
Noise prediction network: ε_θ(x_t, t)

U-Net with modifications:
  - Takes noisy image x_t AND timestep t
  - Timestep embedding (like positional encoding)
  - Self-attention at some resolutions
  - Skip connections for detail preservation

      x_t + time_embedding
              ↓
    ┌─────────────────────┐
    │  Conv + TimeEmbed   │
    └─────────┬───────────┘
              ↓ (downsample)
    ┌─────────────────────┐
    │  ResBlock + Attn    │─────────────┐
    └─────────┬───────────┘             │
              ↓ (downsample)            │
    ┌─────────────────────┐             │
    │      Bottleneck     │             │
    └─────────┬───────────┘             │
              ↓ (upsample)              │
    ┌─────────────────────┐             │
    │  ResBlock + Attn    │←────────────┘
    └─────────┬───────────┘  (skip connection)
              ↓ (upsample)
    ┌─────────────────────┐
    │  Conv → ε_θ         │
    └─────────────────────┘
""")

print("""
Timestep embedding:
  - Similar to transformer positional encoding
  - Sinusoidal or learned
  - Added/concatenated to features at each layer
  
  def timestep_embedding(t, dim):
      half = dim // 2
      freqs = exp(-log(10000) * arange(half) / half)
      args = t * freqs
      return concat([cos(args), sin(args)])
""")
```

## Training Algorithm

```python
print("\n=== TRAINING DDPM ===")
print("""
Algorithm 1: Training

repeat:
    x_0 ~ q(x_0)           # Sample data
    t ~ Uniform(1, T)      # Random timestep
    ε ~ N(0, I)            # Sample noise
    
    x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε   # Add noise
    
    loss = ||ε - ε_θ(x_t, t)||²         # Predict noise
    
    gradient step on loss
until converged

Key insights:
  - Train on all timesteps simultaneously
  - Simple MSE loss on noise prediction
  - Don't need to run full diffusion chain
""")

def training_step(model, x_0, T, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    """One training step (pseudocode)"""
    batch_size = x_0.shape[0]
    
    # Sample random timesteps
    t = np.random.randint(0, T, batch_size)
    
    # Sample noise
    noise = np.random.randn(*x_0.shape)
    
    # Create noisy images
    x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * noise
    
    # Predict noise
    predicted_noise = model(x_t, t)
    
    # MSE loss
    loss = np.mean((noise - predicted_noise) ** 2)
    
    return loss
```

## Sampling (Generation)

```python
print("\n=== SAMPLING FROM DDPM ===")
print("""
Algorithm 2: Sampling

x_T ~ N(0, I)              # Start with pure noise

for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else z = 0
    x_{t-1} = μ_θ(x_t, t) + σ_t × z
    
return x_0

The model predicts noise at each step,
which is used to compute μ_θ.

Sampling is SLOW:
  - Need T forward passes (typically T=1000)
  - Can't parallelize steps
  
Solutions: DDIM, faster samplers
""")

def sample(model, T, shape, alpha, alpha_bar, beta):
    """Generate samples from trained model"""
    # Start with pure noise
    x = np.random.randn(*shape)
    
    for t in reversed(range(T)):
        # Predict noise
        predicted_noise = model(x, t)
        
        # Reverse step
        sqrt_alpha = np.sqrt(alpha[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])
        
        mean = (1 / sqrt_alpha) * (x - (beta[t] / sqrt_one_minus_alpha_bar) * predicted_noise)
        
        if t > 0:
            noise = np.random.randn(*shape)
            x = mean + np.sqrt(beta[t]) * noise
        else:
            x = mean
    
    return x

print("1000 steps is slow! Modern methods use fewer steps:")
print("  DDIM: 50-100 steps with same quality")
print("  DPM-Solver: 10-20 steps")
print("  Consistency Models: 1-2 steps")
```

## DDIM Sampling

```python
print("\n=== DDIM (Faster Sampling) ===")
print("""
DDIM: Denoising Diffusion Implicit Models

Key insight: Non-Markovian process
  - Skip steps while maintaining quality
  - Deterministic option (η=0)

DDIM update:
  x_{t-1} = √ᾱ_{t-1} × predicted_x_0
          + √(1-ᾱ_{t-1}-σ²) × ε_θ(x_t)
          + σ × z
          
where predicted_x_0 = (x_t - √(1-ᾱ_t) × ε_θ) / √ᾱ_t

Benefits:
  - Use 50-100 steps instead of 1000
  - η=0: Deterministic (same noise → same image)
  - η=1: Same as DDPM
  - Interpolate latents for smooth transitions
""")

print("""
Timestep stride example:
  T=1000, use 50 steps
  timesteps = [0, 20, 40, 60, ..., 980]
  
  Quality nearly identical to 1000 steps!
""")
```

## Conditional Generation

```python
print("\n=== CONDITIONAL DIFFUSION ===")
print("""
Generate images matching a condition:

1. CLASSIFIER GUIDANCE:
   - Train separate classifier on noisy images
   - Use gradient to push toward class
   
   ε_guided = ε_θ(x_t, t) - s × ∇_x log p_φ(y|x_t)
   
   s: Guidance scale (higher = more class influence)

2. CLASSIFIER-FREE GUIDANCE:
   - Train unconditional AND conditional model together
   - Interpolate predictions
   
   ε_guided = (1+w) × ε_θ(x_t, t, y) - w × ε_θ(x_t, t, ∅)
   
   w: Guidance weight (typically 3-15)
   ∅: Null/empty condition
   
   During training: Randomly drop condition (e.g., 10%)
   During sampling: Use both conditional and unconditional
""")

print("""
Text-to-image (e.g., Stable Diffusion):
  - Condition on text embedding from CLIP/T5
  - Cross-attention between U-Net and text
  - Classifier-free guidance for quality
  
  "A cat wearing a hat, digital art"
       ↓ (text encoder)
  [embedding] → cross-attention in U-Net
""")
```

## Key Points

- **Forward process**: Gradually add noise (fixed, no learning)
- **Reverse process**: Learn to denoise step by step
- **Noise prediction**: Network predicts ε added at each step
- **U-Net**: Standard architecture with time embedding
- **DDIM**: Faster sampling with fewer steps
- **Classifier-free guidance**: Control generation without classifier

## Reflection Questions

1. Why is predicting noise easier than predicting the clean image directly?
2. How does classifier-free guidance improve generation quality?
3. What's the trade-off between number of sampling steps and quality?
