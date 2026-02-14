# The Transformer Architecture

## Introduction

Transformers replaced recurrence with self-attention, enabling massive parallelization and better long-range dependencies. They power modern NLP models like BERT and GPT.

## Beyond RNNs

```python
import numpy as np
import pandas as pd

print("=== WHY TRANSFORMERS? ===")
print("""
RNN LIMITATIONS:
  1. Sequential processing - can't parallelize
  2. Vanishing gradients over long sequences
  3. Information bottleneck at each step

TRANSFORMER ADVANTAGES:
  1. Parallel processing - all positions at once
  2. Direct connections between any positions
  3. Constant path length for any dependency

RNN: x_1 → x_2 → x_3 → ... → x_100 (100 steps for x_1 to reach x_100)
Transformer: x_1 ↔ x_100 directly! (single attention step)
""")
```

## Transformer Architecture

```python
print("\n=== TRANSFORMER ARCHITECTURE ===")
print("""
Original "Attention Is All You Need" (2017):

ENCODER (6 identical layers):
  Each layer:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual + layer norm)
    3. Feed-Forward Network
    4. Add & Norm

DECODER (6 identical layers):
  Each layer:
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Multi-Head Cross-Attention (to encoder)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm

┌─────────────────────────────────────────┐
│              TRANSFORMER                 │
├───────────────────┬─────────────────────┤
│     ENCODER       │      DECODER        │
├───────────────────┼─────────────────────┤
│                   │  Output Embedding   │
│                   │         ↓           │
│ Multi-Head        │  Masked Multi-Head  │
│ Self-Attention    │  Self-Attention     │
│     ↓             │         ↓           │
│ Add & Norm        │  Add & Norm         │
│     ↓             │         ↓           │
│                   │  Multi-Head         │
│                   │  Cross-Attention←───┤
│                   │         ↓           │
│                   │  Add & Norm         │
│                   │         ↓           │
│ Feed-Forward      │  Feed-Forward       │
│     ↓             │         ↓           │
│ Add & Norm        │  Add & Norm         │
│     ↓             │         ↓           │
│  (×6 layers)      │   (×6 layers)       │
└───────────────────┴─────────────────────┘
""")
```

## Positional Encoding

```python
print("\n=== POSITIONAL ENCODING ===")
print("""
PROBLEM: Self-attention has no notion of position
  - "cat sat mat" same attention as "mat sat cat"
  - Order matters!

SOLUTION: Add positional information to embeddings

Sinusoidal encoding (original paper):
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Properties:
  - Each position has unique encoding
  - Relative positions can be learned
  - Generalizes to longer sequences
""")

def positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encoding"""
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    
    # Compute division term
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sin to even indices, cos to odd
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Example
pe = positional_encoding(10, 8)
print("Positional encoding (first 5 positions, 8 dims):")
print(pe[:5].round(3))
print("\nNote: Different patterns for different positions")
print("Low dimensions: Fast changes (local position)")
print("High dimensions: Slow changes (global position)")
```

## Self-Attention in Detail

```python
print("\n=== SELF-ATTENTION ===")
print("""
Self-attention: Each position attends to ALL positions (including itself)

Input: X (seq_len × d_model)

1. Create Q, K, V:
   Q = X × W_Q
   K = X × W_K
   V = X × W_V

2. Compute attention:
   Attention = softmax(Q × K^T / √d_k) × V

Each position:
  - Query: "What am I looking for?"
  - Key: "What do I have?"
  - Value: "What do I provide?"

Attention score between positions i and j:
  score_ij = q_i · k_j / √d_k
  
High score = j is relevant to i
""")

def self_attention_example():
    """Demonstrate self-attention computation"""
    # Simple 3-word sentence, 4-dim embeddings
    X = np.array([
        [1, 0, 0, 1],   # "The"
        [0, 1, 1, 0],   # "cat"
        [1, 1, 0, 0]    # "sat"
    ])
    
    # Weight matrices (simplified)
    np.random.seed(42)
    d_k = 4
    W_Q = np.random.randn(4, d_k) * 0.5
    W_K = np.random.randn(4, d_k) * 0.5
    W_V = np.random.randn(4, 4) * 0.5
    
    # Compute Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Softmax
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    print("Attention weights:")
    print("          The   cat   sat")
    for i, word in enumerate(['The', 'cat', 'sat']):
        print(f"  {word:4}: {weights[i].round(3)}")
    
    return weights

self_attention_example()
```

## Masked Self-Attention

```python
print("\n=== MASKED SELF-ATTENTION (Decoder) ===")
print("""
In decoder, can't look at future tokens!

"The cat sat on the" → predicting "mat"
  - Can see: "The", "cat", "sat", "on", "the"
  - Cannot see: "mat" (future)

CAUSAL MASK:
  position i can only attend to positions 0, 1, ..., i

         The  cat  sat  on   the  mat
The      ✓    ✗    ✗    ✗    ✗    ✗
cat      ✓    ✓    ✗    ✗    ✗    ✗
sat      ✓    ✓    ✓    ✗    ✗    ✗
on       ✓    ✓    ✓    ✓    ✗    ✗
the      ✓    ✓    ✓    ✓    ✓    ✗
mat      ✓    ✓    ✓    ✓    ✓    ✓

✓ = can attend, ✗ = masked (set to -∞ before softmax)
""")

def create_causal_mask(seq_len):
    """Create causal (look-ahead) mask"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask == 0  # True where we CAN attend

def masked_attention(Q, K, V, mask):
    """Attention with causal mask"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask: -inf where mask is False
    scores = np.where(mask, scores, -1e9)
    
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return weights

# Example
mask = create_causal_mask(4)
print("Causal mask (True = can attend):")
print(mask.astype(int))
```

## Feed-Forward Network

```python
print("\n=== FEED-FORWARD NETWORK ===")
print("""
Position-wise feed-forward network:

FFN(x) = ReLU(x × W_1 + b_1) × W_2 + b_2

Or with GELU (modern):
FFN(x) = GELU(x × W_1) × W_2

Dimensions:
  Input: d_model (512)
  Hidden: d_ff (2048) - typically 4× d_model
  Output: d_model (512)

Applied independently to each position.
Gives model non-linear transformation capacity.
""")

def feed_forward(x, W1, W2, b1, b2):
    """Position-wise feed-forward network"""
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    output = hidden @ W2 + b2
    return output

print("FFN expands then contracts:")
print("  512 → 2048 → 512")
print("  This 'bottleneck' forces learning compressed representations")
```

## Residual Connections & Layer Norm

```python
print("\n=== RESIDUAL CONNECTIONS & LAYER NORM ===")
print("""
RESIDUAL CONNECTIONS:
  output = LayerNorm(x + Sublayer(x))
  
Benefits:
  - Gradients flow directly through additions
  - Easier to learn identity function
  - Enables very deep networks

LAYER NORMALIZATION:
  Normalize across features (not batch):
  
  LayerNorm(x) = γ × (x - μ) / σ + β
  
  μ, σ computed per sample, across features
  γ, β are learned scale and shift

Why Layer Norm (not Batch Norm)?
  - Works with variable sequence lengths
  - Same normalization at train and test time
  - No dependency on batch size
""")

def layer_norm(x, gamma, beta, eps=1e-6):
    """Layer normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta

# Example
x = np.random.randn(3, 4)  # 3 positions, 4 features
gamma = np.ones(4)
beta = np.zeros(4)

normalized = layer_norm(x, gamma, beta)
print("Before LayerNorm (first position):", x[0].round(3))
print("After LayerNorm (first position):", normalized[0].round(3))
print(f"Mean: {normalized[0].mean():.6f}, Std: {normalized[0].std():.3f}")
```

## Putting It Together

```python
print("\n=== TRANSFORMER ENCODER LAYER ===")
print("""
def encoder_layer(x, self_attn, ffn, norm1, norm2):
    # Self-attention block
    attn_output = self_attn(x, x, x)  # Q=K=V=x
    x = norm1(x + attn_output)        # Residual + norm
    
    # Feed-forward block
    ffn_output = ffn(x)
    x = norm2(x + ffn_output)         # Residual + norm
    
    return x

def transformer_encoder(x, layers, pe):
    # Add positional encoding
    x = x + pe[:x.shape[0]]
    
    # Pass through encoder layers
    for layer in layers:
        x = layer(x)
    
    return x
""")

print("""
TRANSFORMER DECODER LAYER:

def decoder_layer(x, enc_output, self_attn, cross_attn, ffn, ...):
    # Masked self-attention
    attn_output = self_attn(x, x, x, causal_mask)
    x = norm1(x + attn_output)
    
    # Cross-attention to encoder
    cross_output = cross_attn(x, enc_output, enc_output)
    x = norm2(x + cross_output)
    
    # Feed-forward
    ffn_output = ffn(x)
    x = norm3(x + ffn_output)
    
    return x
""")
```

## Key Points

- **No recurrence**: Parallelizable, handles long-range deps
- **Positional encoding**: Injects position information
- **Self-attention**: Each position attends to all positions
- **Causal mask**: Decoder can't see future tokens
- **Residual + LayerNorm**: Enables deep networks
- **Multi-head**: Multiple attention patterns in parallel

## Reflection Questions

1. Why does removing recurrence enable parallelization during training?
2. What would happen without positional encoding?
3. Why does the decoder need both masked self-attention and cross-attention?
