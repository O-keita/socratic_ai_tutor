# Recurrent Neural Networks (RNNs)

## Introduction

Recurrent Neural Networks process sequential data by maintaining a hidden state that carries information across time steps. They're fundamental for language, time series, and any ordered data.

## Why RNNs for Sequences?

```python
import numpy as np
import pandas as pd

print("=== RECURRENT NEURAL NETWORKS ===")
print("""
PROBLEM: Standard neural networks can't handle sequences

Feedforward NN:
  - Fixed input size
  - No memory of previous inputs
  - Each input processed independently

SEQUENCES need:
  - Variable length handling
  - Context from previous elements
  - Order matters!

Examples: Text, Speech, Time series, Video, DNA
""")

print("""
RNN key idea: HIDDEN STATE

The hidden state acts as "memory"
  - Summarizes information seen so far
  - Passed from one step to next
  - Updated at each time step

h_t = f(h_{t-1}, x_t)
  h_t: Current hidden state
  h_{t-1}: Previous hidden state
  x_t: Current input
""")
```

## RNN Architecture

```python
print("\n=== RNN ARCHITECTURE ===")
print("""
Unrolled RNN:

    h_0 ──→ h_1 ──→ h_2 ──→ h_3 ──→ h_4
            ↑        ↑        ↑        ↑
           x_1      x_2      x_3      x_4
           
At each step:
  h_t = tanh(W_h × h_{t-1} + W_x × x_t + b)
  
Same weights W_h, W_x used at every step!
  - Parameter sharing across time
  - Handles variable length
  - Learns patterns regardless of position
""")

def rnn_forward(inputs, h_0, W_h, W_x, b):
    """Simple RNN forward pass"""
    h = h_0
    hidden_states = []
    
    for x in inputs:
        # RNN cell
        h = np.tanh(np.dot(W_h, h) + np.dot(W_x, x) + b)
        hidden_states.append(h)
    
    return hidden_states

# Example
np.random.seed(42)
hidden_dim = 4
input_dim = 3
seq_length = 5

W_h = np.random.randn(hidden_dim, hidden_dim) * 0.1
W_x = np.random.randn(hidden_dim, input_dim) * 0.1
b = np.zeros(hidden_dim)

# Sequence of inputs
inputs = [np.random.randn(input_dim) for _ in range(seq_length)]
h_0 = np.zeros(hidden_dim)

hidden_states = rnn_forward(inputs, h_0, W_h, W_x, b)

print("Hidden states through time:")
for t, h in enumerate(hidden_states):
    print(f"  t={t}: {h.round(3)}")
```

## Output Modes

```python
print("\n=== OUTPUT MODES ===")
print("""
1. MANY-TO-ONE (Classification/Regression)
   Input: Sequence → Output: Single value
   Example: Sentiment analysis
   
   [The] [movie] [was] [great] → Positive
   
   Use final hidden state for prediction.

2. MANY-TO-MANY (Sequence Labeling)
   Input: Sequence → Output: Sequence (same length)
   Example: POS tagging, NER
   
   [The] [cat] [sat] → [DET] [NOUN] [VERB]

3. MANY-TO-MANY (Seq2Seq, different length)
   Input: Sequence → Output: Different length sequence
   Example: Translation
   
   [Hello] [world] → [Bonjour] [le] [monde]
   
   Encoder-decoder architecture.

4. ONE-TO-MANY (Generation)
   Input: Single → Output: Sequence
   Example: Image captioning
""")
```

## Vanishing Gradient Problem

```python
print("\n=== VANISHING GRADIENT PROBLEM ===")
print("""
PROBLEM: Gradients vanish during backpropagation through time

When training RNN:
  - Gradients multiply through each time step
  - If multiplied values < 1, gradients → 0
  - Early time steps get tiny gradients
  - Network forgets long-ago information

Example: "The cat, which had been sleeping 
          on the warm sunny windowsill, was..."
          
By "was", RNN has forgotten "cat" is singular.
""")

# Demonstrate gradient decay
print("Gradient magnitude through time steps:")
gradient = 1.0
decay_factor = 0.9

for t in range(15):
    print(f"  t={t:2d}: gradient = {gradient:.6f}")
    gradient *= decay_factor

print(f"\nAfter 15 steps, gradient is only {gradient:.6f} of original!")
print("Early layers barely learn.")
```

## LSTM: Long Short-Term Memory

```python
print("\n=== LSTM ===")
print("""
LSTM solves vanishing gradients with GATES:

1. FORGET GATE: What to forget from cell state
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   
2. INPUT GATE: What new info to add
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
   
3. CELL STATE UPDATE:
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
   
4. OUTPUT GATE: What to output
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(c_t)

Key innovation: Cell state c_t
  - Information highway through time
  - Gradients flow unchanged (when forget gate ≈ 1)
  - Gates learn what to remember/forget
""")

def lstm_cell(x, h_prev, c_prev, weights):
    """Simplified LSTM cell"""
    W_f, W_i, W_c, W_o = weights
    b_f, b_i, b_c, b_o = [np.zeros(h_prev.shape) for _ in range(4)]
    
    combined = np.concatenate([h_prev, x])
    
    # Gates
    f = 1 / (1 + np.exp(-np.dot(W_f, combined)))  # Forget
    i = 1 / (1 + np.exp(-np.dot(W_i, combined)))  # Input
    o = 1 / (1 + np.exp(-np.dot(W_o, combined)))  # Output
    
    # Cell state
    c_tilde = np.tanh(np.dot(W_c, combined))
    c = f * c_prev + i * c_tilde
    
    # Hidden state
    h = o * np.tanh(c)
    
    return h, c

print("LSTM maintains TWO states:")
print("  - Cell state (c): Long-term memory")
print("  - Hidden state (h): Short-term/output")
```

## GRU: Gated Recurrent Unit

```python
print("\n=== GRU ===")
print("""
GRU: Simplified LSTM with 2 gates (instead of 3)

1. RESET GATE: How much past to forget
   r_t = σ(W_r · [h_{t-1}, x_t])
   
2. UPDATE GATE: How much to update
   z_t = σ(W_z · [h_{t-1}, x_t])

Hidden state update:
   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Comparison to LSTM:
  - Fewer parameters (2 gates vs 3)
  - No separate cell state
  - Often similar performance
  - Faster to train
""")

def gru_cell(x, h_prev, W_r, W_z, W):
    """Simplified GRU cell"""
    combined = np.concatenate([h_prev, x])
    
    # Gates
    r = 1 / (1 + np.exp(-np.dot(W_r, combined)))  # Reset
    z = 1 / (1 + np.exp(-np.dot(W_z, combined)))  # Update
    
    # Candidate
    combined_reset = np.concatenate([r * h_prev, x])
    h_tilde = np.tanh(np.dot(W, combined_reset))
    
    # New hidden state
    h = (1 - z) * h_prev + z * h_tilde
    
    return h

print("GRU vs LSTM:")
print("  LSTM: 4 weight matrices, 2 states (h, c)")
print("  GRU: 3 weight matrices, 1 state (h)")
```

## Bidirectional RNNs

```python
print("\n=== BIDIRECTIONAL RNNs ===")
print("""
Process sequence in BOTH directions:

Forward:  x_1 → x_2 → x_3 → x_4 → x_5
              ↓     ↓     ↓     ↓
Backward: x_1 ← x_2 ← x_3 ← x_4 ← x_5

Each position gets TWO hidden states:
  h_t = [h_forward_t ; h_backward_t]

Benefits:
  - Full context at each position
  - Past AND future information
  - Better for classification/labeling

Limitation:
  - Cannot use for generation (no future yet)
  
Example: NER
  "Apple announced new iPhone"
  - Forward: "Apple" might be fruit or company
  - Backward: Seeing "iPhone" clarifies it's the company
""")
```

## Keras Implementation

```python
print("\n=== KERAS RNN LAYERS ===")
print("""
# Simple RNN
model.add(SimpleRNN(64, return_sequences=True))

# LSTM
model.add(LSTM(64, return_sequences=True))

# GRU
model.add(GRU(64, return_sequences=True))

# Bidirectional
model.add(Bidirectional(LSTM(64, return_sequences=True)))

Key parameters:
  - units: Hidden state dimension
  - return_sequences: Output all timesteps (True) or just last (False)
  - return_state: Also return final state(s)
  
Stacking RNNs:
  model.add(LSTM(64, return_sequences=True))  # Need sequences for next layer
  model.add(LSTM(32, return_sequences=False)) # Last layer, single output

Text classification example:

model = Sequential([
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
""")
```

## Key Points

- **RNNs**: Process sequences with hidden state memory
- **Vanishing gradients**: Long-range dependencies hard to learn
- **LSTM**: Gates control information flow, solves vanishing gradient
- **GRU**: Simpler than LSTM, often similar performance
- **Bidirectional**: Sees past and future context
- **return_sequences**: True for seq2seq, False for classification

## Reflection Questions

1. Why does parameter sharing across time steps enable variable-length sequences?
2. How do LSTM gates help information persist over long sequences?
3. When would you choose GRU over LSTM?
