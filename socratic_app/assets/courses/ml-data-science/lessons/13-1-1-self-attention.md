# Transformers: Self-Attention

## Introduction

The Transformer architecture revolutionized NLP and beyond. At its heart is the self-attention mechanism, which allows models to weigh the importance of different parts of the input.

## Core Concepts

### What is Self-Attention?

Self-attention computes relationships between all positions in a sequence:

For each word, ask: "How relevant is every other word to understanding me?"

### The Attention Formula

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Where:
- Q (Query): What am I looking for?
- K (Key): What do I have to offer?
- V (Value): What information do I contain?
- $d_k$: Key dimension (for scaling)

### Computing Attention Step by Step

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Compute attention scores
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # Step 2: Apply softmax to get attention weights
    weights = softmax(scores)
    
    # Step 3: Weighted sum of values
    output = np.matmul(weights, V)
    
    return output, weights
```

### Example: "The cat sat on the mat"

```python
# Simplified example
# Each word represented as embedding

words = ["The", "cat", "sat", "on", "the", "mat"]

# After computing attention for "sat":
# High weight: "cat" (who sat?)
# High weight: "mat" (where?)
# Lower weight: "The", "on"
```

### Creating Q, K, V

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Linear projections
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Compute attention                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        return self.fc_out(out)
```

### Why Self-Attention Works

1. **Captures long-range dependencies**: Unlike RNNs, direct connections between any positions
2. **Parallelizable**: All positions computed simultaneously
3. **Interpretable**: Attention weights show what the model "looks at"

### Attention Visualization

```python
# Attention matrix shows word relationships
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(attention_weights, 
            xticklabels=words,
            yticklabels=words,
            cmap='Blues')
plt.title("Self-Attention Weights")
```

---

## Key Points

- Self-attention computes pairwise relationships in sequences
- Q, K, V are learned linear projections of input
- Scaling by √d_k prevents small gradients from softmax
- Enables parallel computation (vs sequential RNNs)
- Attention weights are interpretable

---

## Reflection Questions

1. **Think**: Why scale by √d_k? What happens to softmax with large dot products?

2. **Consider**: How does self-attention compare to RNNs for capturing long-range dependencies? What's the computational tradeoff?

3. **Explore**: In the sentence "The animal didn't cross the street because it was too tired", how might attention help resolve what "it" refers to?
