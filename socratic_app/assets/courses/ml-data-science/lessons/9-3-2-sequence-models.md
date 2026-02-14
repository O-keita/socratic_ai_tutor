# Sequence Models for NLP

## Introduction

Sequence models process text as ordered sequences of words, capturing temporal dependencies and context that bag-of-words approaches miss.

## Why Sequence Matters

```python
import numpy as np
import pandas as pd

print("=== SEQUENCE MODELS FOR NLP ===")
print("""
WHY SEQUENCE ORDER MATTERS:

Bag-of-words loses order:
  "dog bites man" = "man bites dog" (same words!)
  
But meaning is completely different!

Sequence models:
  - Process text word by word
  - Maintain hidden state
  - Capture dependencies between words
  
Applications:
  - Language modeling (predict next word)
  - Machine translation
  - Text generation
  - Named entity recognition
  - Sentiment analysis with context
""")
```

## Recurrent Neural Networks

```python
print("\n=== RECURRENT NEURAL NETWORKS ===")
print("""
RNN: Processes sequences one element at a time

At each step t:
  h_t = f(W_h × h_{t-1} + W_x × x_t + b)
  
Where:
  x_t = current input (word embedding)
  h_{t-1} = previous hidden state
  h_t = new hidden state
  
The hidden state acts as "memory" of past inputs.

Architecture:
  [word_1] → [RNN] → h_1
                ↓
  [word_2] → [RNN] → h_2
                ↓
  [word_3] → [RNN] → h_3 → [output]
""")

# Simple RNN forward pass
def simple_rnn_cell(x, h_prev, W_h, W_x, b):
    """Single RNN cell computation"""
    return np.tanh(np.dot(W_h, h_prev) + np.dot(W_x, x) + b)

# Example
hidden_size = 4
input_size = 3

np.random.seed(42)
W_h = np.random.randn(hidden_size, hidden_size) * 0.1
W_x = np.random.randn(hidden_size, input_size) * 0.1
b = np.zeros(hidden_size)

# Process sequence
sequence = [
    np.random.randn(input_size),  # "The"
    np.random.randn(input_size),  # "cat"
    np.random.randn(input_size),  # "sat"
]

h = np.zeros(hidden_size)  # Initial hidden state
print("RNN processing sequence:")
for t, x in enumerate(sequence):
    h = simple_rnn_cell(x, h, W_h, W_x, b)
    print(f"  Step {t}: h = {h.round(3)}")
```

## Vanishing Gradient Problem

```python
print("\n=== VANISHING GRADIENT PROBLEM ===")
print("""
RNNs struggle with long sequences:

During backpropagation through time:
  - Gradients get multiplied repeatedly
  - If weights < 1: gradients vanish → 0
  - If weights > 1: gradients explode → ∞

Impact:
  - Hard to learn long-range dependencies
  - "The cat that sat on the mat WAS" 
  - RNN forgets "cat" by the time it reaches "was"

Solutions:
  1. LSTM (Long Short-Term Memory)
  2. GRU (Gated Recurrent Unit)
  3. Gradient clipping
  4. Skip connections
""")

# Demonstrate gradient decay
print("\nGradient magnitude over 10 steps:")
gradient = 1.0
weight = 0.9  # < 1 causes vanishing
for t in range(10):
    gradient *= weight
    print(f"  Step {t}: gradient = {gradient:.6f}")
print("\nGradient nearly vanishes!")
```

## Long Short-Term Memory (LSTM)

```python
print("\n=== LSTM ===")
print("""
LSTM adds gates to control information flow:

1. FORGET GATE: What to forget from cell state
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   
2. INPUT GATE: What new info to store
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   
3. CELL STATE UPDATE:
   C_t = f_t * C_{t-1} + i_t * tanh(W_c · [h_{t-1}, x_t] + b_c)
   
4. OUTPUT GATE: What to output
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)

Key innovation:
  - Cell state acts as "highway" for gradients
  - Gates learn what to remember/forget
  - Can maintain information over long sequences
""")

def lstm_cell(x, h_prev, c_prev, weights):
    """Simplified LSTM cell (illustrative)"""
    W_f, W_i, W_c, W_o = weights['W_f'], weights['W_i'], weights['W_c'], weights['W_o']
    
    # Concatenate input and previous hidden state
    combined = np.concatenate([h_prev, x])
    
    # Gates
    f = 1 / (1 + np.exp(-np.dot(W_f, combined)))  # Forget gate
    i = 1 / (1 + np.exp(-np.dot(W_i, combined)))  # Input gate
    o = 1 / (1 + np.exp(-np.dot(W_o, combined)))  # Output gate
    
    # Candidate cell state
    c_candidate = np.tanh(np.dot(W_c, combined))
    
    # New cell state
    c = f * c_prev + i * c_candidate
    
    # New hidden state
    h = o * np.tanh(c)
    
    return h, c, {'forget': f, 'input': i, 'output': o}

print("LSTM maintains separate cell state and hidden state")
print("Gates values are between 0 and 1 (sigmoid)")
```

## Gated Recurrent Unit (GRU)

```python
print("\n=== GRU ===")
print("""
GRU: Simplified LSTM with fewer parameters

Only 2 gates:
1. RESET GATE: How much past info to forget
   r_t = σ(W_r · [h_{t-1}, x_t])
   
2. UPDATE GATE: How much to update hidden state
   z_t = σ(W_z · [h_{t-1}, x_t])

Hidden state:
   h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
   h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

Comparison:
  LSTM: 4 gates, cell state + hidden state
  GRU: 2 gates, only hidden state
  
GRU is faster, often comparable performance.
""")

def gru_cell(x, h_prev, W_r, W_z, W):
    """Simplified GRU cell"""
    combined = np.concatenate([h_prev, x])
    
    # Reset gate
    r = 1 / (1 + np.exp(-np.dot(W_r, combined)))
    
    # Update gate  
    z = 1 / (1 + np.exp(-np.dot(W_z, combined)))
    
    # Candidate hidden state
    combined_reset = np.concatenate([r * h_prev, x])
    h_candidate = np.tanh(np.dot(W, combined_reset))
    
    # New hidden state
    h = (1 - z) * h_prev + z * h_candidate
    
    return h
```

## Bidirectional RNNs

```python
print("\n=== BIDIRECTIONAL RNNs ===")
print("""
Process sequence in BOTH directions:

Forward:  [w1] → [w2] → [w3] → [w4]
Backward: [w1] ← [w2] ← [w3] ← [w4]

Final representation combines both:
  h_t = [h_forward_t ; h_backward_t]

Benefits:
  - Each word sees full context (past + future)
  - Better for tasks like NER, classification
  
Cannot use for:
  - Real-time generation (don't have future)
  - Language modeling
""")

def bidirectional_process(sequence, forward_rnn, backward_rnn):
    """Process sequence bidirectionally"""
    # Forward pass
    h_forward = []
    h = np.zeros(4)  # Initial hidden state
    for x in sequence:
        h = simple_rnn_cell(x, h, W_h, W_x, b)
        h_forward.append(h)
    
    # Backward pass
    h_backward = []
    h = np.zeros(4)
    for x in reversed(sequence):
        h = simple_rnn_cell(x, h, W_h, W_x, b)
        h_backward.insert(0, h)
    
    # Concatenate
    h_combined = [np.concatenate([f, b]) for f, b in zip(h_forward, h_backward)]
    return h_combined

print("Example bidirectional output shape:")
print(f"  Forward hidden: {hidden_size}")
print(f"  Backward hidden: {hidden_size}")
print(f"  Combined: {hidden_size * 2}")
```

## Sequence Labeling Architecture

```python
print("\n=== SEQUENCE LABELING ===")
print("""
BiLSTM for tasks like NER, POS tagging:

Architecture:
  Embedding → BiLSTM → Dense → Labels

Each token gets a label:
  [The]  → [O]
  [Apple] → [B-ORG]
  [Inc]  → [I-ORG]
  [stock] → [O]

Training:
  - Cross-entropy loss at each position
  - Backprop through time (BPTT)
  
Often combined with CRF layer:
  BiLSTM-CRF: Better label dependencies
""")

# Pseudo-code architecture
print("""
PyTorch-style architecture:

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits
""")
```

## Text Classification with RNN

```python
print("\n=== TEXT CLASSIFICATION WITH RNN ===")
print("""
Use final hidden state for classification:

Sentence: "This movie was great"
  [This] → h1
  [movie] → h2  
  [was] → h3
  [great] → h4 → [Dense] → [positive/negative]

Alternatives:
  1. Use last hidden state
  2. Use max pooling over all states
  3. Use attention-weighted sum
  4. Concatenate [first; last] states
""")

def rnn_classifier_forward(sentence_embeddings, W_h, W_x, b, W_out):
    """RNN text classifier forward pass"""
    h = np.zeros(hidden_size)
    
    # Process sequence
    for x in sentence_embeddings:
        h = simple_rnn_cell(x, h, W_h, W_x, b)
    
    # Classification from final hidden state
    logits = np.dot(W_out, h)
    
    # Softmax
    probs = np.exp(logits) / np.exp(logits).sum()
    
    return probs

print("Final hidden state summarizes entire sequence")
```

## Key Points

- **RNNs**: Process sequences maintaining hidden state memory
- **Vanishing gradients**: Limit RNN's long-range memory
- **LSTM**: Gates control what to remember/forget
- **GRU**: Simplified LSTM, often comparable performance
- **Bidirectional**: Sees both past and future context
- **Sequence labeling**: BiLSTM outputs label per token
- **Classification**: Use final hidden state or pooling

## Reflection Questions

1. Why can't bidirectional models be used for text generation?
2. How do LSTM gates help with the vanishing gradient problem?
3. When might you prefer GRU over LSTM?
