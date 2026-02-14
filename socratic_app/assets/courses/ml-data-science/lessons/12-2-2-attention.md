# Attention Mechanisms

## Introduction

Attention allows models to focus on relevant parts of the input when producing each output. It revolutionized sequence modeling by removing the context bottleneck.

## The Attention Intuition

```python
import numpy as np
import pandas as pd

print("=== ATTENTION INTUITION ===")
print("""
PROBLEM: Fixed-size context loses information

Translation: "The cat sat on the mat" → "Le chat était assis sur le tapis"

For generating "chat":
  - Most relevant: "cat"
  - Less relevant: "mat", "sat"
  
SOLUTION: Look at all encoder states, weighted by relevance

At each decoder step:
  1. Score how relevant each encoder state is
  2. Convert scores to weights (softmax)
  3. Weighted sum of encoder states
  4. Use this "context" for prediction
""")
```

## Attention Mechanism

```python
print("\n=== HOW ATTENTION WORKS ===")
print("""
Components:
  - Query (Q): What we're looking for (decoder state)
  - Keys (K): What we're searching through (encoder states)  
  - Values (V): What we retrieve (encoder states)

Steps:
  1. SCORE: Compare query with each key
     score_i = f(query, key_i)
     
  2. NORMALIZE: Softmax to get weights
     weights = softmax(scores)
     
  3. AGGREGATE: Weighted sum of values
     context = Σ weight_i × value_i

Different scoring functions:
  - Dot product: Q · K
  - Scaled dot product: (Q · K) / √d
  - Additive: v × tanh(W_q × Q + W_k × K)
""")

def attention(query, keys, values):
    """Simple attention mechanism"""
    # Score: dot product
    scores = np.dot(keys, query)
    
    # Normalize: softmax
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Aggregate: weighted sum
    context = np.sum(values * weights[:, np.newaxis], axis=0)
    
    return context, weights

# Example
np.random.seed(42)
seq_length = 4
hidden_dim = 3

# Encoder outputs (keys and values are same in basic attention)
encoder_states = np.random.randn(seq_length, hidden_dim)
decoder_state = np.random.randn(hidden_dim)  # Query

context, weights = attention(decoder_state, encoder_states, encoder_states)

print("Encoder states (keys/values):")
print(encoder_states.round(2))
print(f"\nDecoder state (query): {decoder_state.round(2)}")
print(f"\nAttention weights: {weights.round(3)}")
print(f"Context vector: {context.round(3)}")
```

## Scaled Dot-Product Attention

```python
print("\n=== SCALED DOT-PRODUCT ATTENTION ===")
print("""
Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Why scale by √d_k?
  - Large d_k → large dot products
  - Large values → softmax very peaked
  - Gradient vanishes for non-maximum values
  
Scaling keeps gradients healthy.
""")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: queries (batch, seq_len, d_k)
    K: keys (batch, seq_len, d_k)
    V: values (batch, seq_len, d_v)
    """
    d_k = K.shape[-1]
    
    # Scores: Q × K^T / √d_k
    scores = np.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
    
    # Optional masking (for decoder)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    # Softmax
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Weighted sum
    output = np.matmul(weights, V)
    
    return output, weights

# Example
Q = np.random.randn(1, 3, 4)  # 1 batch, 3 queries, dim 4
K = np.random.randn(1, 5, 4)  # 5 keys
V = np.random.randn(1, 5, 6)  # 5 values, dim 6

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Query shape: {Q.shape}")
print(f"Key shape: {K.shape}")
print(f"Value shape: {V.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

## Attention in Seq2Seq

```python
print("\n=== ATTENTION IN SEQ2SEQ ===")
print("""
Encoder produces hidden states: h_1, h_2, ..., h_n

At each decoder step t:
  1. Decoder hidden state s_t is the query
  2. Encoder states are keys and values
  3. Compute attention weights
  4. Get context: c_t = Σ α_ti × h_i
  5. Combine: [s_t; c_t] → prediction

         h1    h2    h3    h4     (encoder outputs)
          ↓     ↓     ↓     ↓
         α1    α2    α3    α4     (attention weights)
          ↘     ↓     ↓     ↙
             context c_t
                  ↓
          [s_t; c_t] → y_t        (prediction)

The context changes at each decoder step!
""")

def attention_seq2seq_step(decoder_state, encoder_outputs):
    """One step of attention-based decoding"""
    # Compute attention
    context, weights = attention(decoder_state, encoder_outputs, encoder_outputs)
    
    # Concatenate decoder state and context
    combined = np.concatenate([decoder_state, context])
    
    return combined, weights

# Simulate translation step
encoder_outputs = np.random.randn(6, 8)  # 6 words, 8-dim
decoder_state = np.random.randn(8)

combined, weights = attention_seq2seq_step(decoder_state, encoder_outputs)

print("Input sentence: 'The cat sat on the mat'")
print(f"Attention weights: {weights.round(3)}")
print(f"Most attended position: {np.argmax(weights)} ('{['The','cat','sat','on','the','mat'][np.argmax(weights)]}')")
```

## Types of Attention

```python
print("\n=== TYPES OF ATTENTION ===")
print("""
1. GLOBAL ATTENTION
   - Attend to all encoder positions
   - Most common approach
   
2. LOCAL ATTENTION
   - Attend to window around aligned position
   - Faster, but needs alignment estimation

3. SELF-ATTENTION
   - Query, key, value from SAME sequence
   - Each position attends to all positions
   - Foundation of Transformers

4. CROSS-ATTENTION
   - Query from one sequence
   - Key/value from another sequence
   - Used in encoder-decoder

5. MULTI-HEAD ATTENTION
   - Multiple attention heads in parallel
   - Each head learns different relationships
   - Concatenate and project
""")

print("""
Self-attention example:
  "The animal didn't cross the street because it was too tired"
  
  What does "it" refer to?
  
  Self-attention allows each word to look at all other words.
  "it" would attend strongly to "animal".
""")
```

## Multi-Head Attention

```python
print("\n=== MULTI-HEAD ATTENTION ===")
print("""
Multiple attention "heads" in parallel:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)

Why multiple heads?
  - Different aspects of relationships
  - Head 1: syntactic dependency
  - Head 2: semantic similarity
  - Head 3: positional relationship
  - etc.
""")

def multi_head_attention(Q, K, V, num_heads=8, d_model=64):
    """Simplified multi-head attention"""
    d_k = d_model // num_heads
    
    heads = []
    for _ in range(num_heads):
        # Project Q, K, V (simplified - random projections)
        W_Q = np.random.randn(d_model, d_k)
        W_K = np.random.randn(d_model, d_k)
        W_V = np.random.randn(d_model, d_k)
        
        Q_proj = np.dot(Q, W_Q)
        K_proj = np.dot(K, W_K)
        V_proj = np.dot(V, W_V)
        
        # Scaled dot-product attention
        scores = np.dot(Q_proj, K_proj.T) / np.sqrt(d_k)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        head = np.dot(weights, V_proj)
        
        heads.append(head)
    
    # Concatenate heads
    concat = np.concatenate(heads, axis=-1)
    
    # Final projection
    W_O = np.random.randn(d_model, d_model)
    output = np.dot(concat, W_O)
    
    return output

print("Multi-head splits attention into parallel streams:")
print(f"  d_model = 512, num_heads = 8")
print(f"  Each head dimension: 512 / 8 = 64")
print(f"  After concat: 8 × 64 = 512 back to d_model")
```

## Attention Visualization

```python
print("\n=== ATTENTION VISUALIZATION ===")
print("""
Attention weights show what the model "looks at":

Translation: "The cat sat on the mat" → "Le chat assis sur le tapis"

              The  cat  sat   on  the  mat
Le            0.6  0.1  0.1  0.1  0.05 0.05
chat          0.1  0.7  0.05 0.05 0.05 0.05
assis         0.05 0.1  0.7  0.05 0.05 0.05
sur           0.05 0.05 0.05 0.7  0.05 0.1
le            0.3  0.1  0.05 0.1  0.4  0.05
tapis         0.05 0.05 0.05 0.1  0.1  0.7

Reading: Each row shows what input words the output word attends to.
"chat" (cat) strongly attends to "cat" (0.7)
""")

# Create sample attention matrix
source = ['The', 'cat', 'sat', 'on', 'the', 'mat']
target = ['Le', 'chat', 'assis', 'sur', 'le', 'tapis']

attention_matrix = np.array([
    [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
    [0.1, 0.7, 0.05, 0.05, 0.05, 0.05],
    [0.05, 0.1, 0.7, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.7, 0.05, 0.1],
    [0.3, 0.1, 0.05, 0.1, 0.4, 0.05],
    [0.05, 0.05, 0.05, 0.1, 0.1, 0.7]
])

print("\nAlignment summary:")
for i, tgt in enumerate(target):
    max_idx = np.argmax(attention_matrix[i])
    print(f"  '{tgt}' ← '{source[max_idx]}' (weight: {attention_matrix[i, max_idx]:.2f})")
```

## Key Points

- **Attention**: Focus on relevant parts of input
- **Query-Key-Value**: Search mechanism for relevance
- **Scaling**: √d_k prevents gradient issues
- **Multi-head**: Multiple parallel attention patterns
- **Self-attention**: Same sequence as Q, K, V
- **Cross-attention**: Different sequences for Q vs K, V

## Reflection Questions

1. How does attention solve the context bottleneck problem?
2. Why use multiple heads instead of one larger attention?
3. What's the difference between self-attention and cross-attention?
