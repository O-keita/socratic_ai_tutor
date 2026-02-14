# Sequence-to-Sequence Models

## Introduction

Sequence-to-sequence (Seq2Seq) models transform one sequence into another of potentially different length. They power machine translation, summarization, and conversational AI.

## Encoder-Decoder Architecture

```python
import numpy as np
import pandas as pd

print("=== ENCODER-DECODER ===")
print("""
Seq2Seq = Encoder + Decoder

ENCODER:
  - Processes input sequence
  - Produces context representation
  - "Understanding" the input

DECODER:
  - Generates output sequence
  - Conditioned on encoder context
  - "Producing" the output

         Encoder                    Decoder
   [x1]→[x2]→[x3]→[x4]  →  context  →  [y1]→[y2]→[y3]
   
The context vector summarizes the entire input.
""")
```

## Basic Seq2Seq

```python
print("\n=== BASIC SEQ2SEQ ===")
print("""
Simple Encoder-Decoder:

ENCODER:
  - RNN/LSTM processes input
  - Final hidden state = context vector
  
DECODER:
  - Initialized with context vector
  - Generates output word by word
  - Each output fed as next input

Problem: Context is a FIXED-SIZE vector
  - Bottleneck for long sequences
  - All information squeezed into one vector
  - Early words get "forgotten"
  
Solution: ATTENTION mechanism (later lesson)
""")

def simple_encoder(input_sequence, hidden_size):
    """Encode input sequence into context"""
    # In practice: RNN/LSTM processes sequence
    # Returns final hidden state as context
    
    print("Encoder processing:")
    for i, token in enumerate(input_sequence):
        print(f"  Step {i}: '{token}'")
    
    # Context is final hidden state
    context = np.random.randn(hidden_size)
    print(f"  → Context vector: {hidden_size}-dimensional")
    return context

def simple_decoder(context, max_length, vocab):
    """Decode context into output sequence"""
    print("\nDecoder generating:")
    
    output = []
    prev_token = '<START>'
    
    for i in range(max_length):
        # In practice: RNN takes context + prev_token
        # Here we just simulate
        next_token = np.random.choice(vocab)
        output.append(next_token)
        print(f"  Step {i}: '{next_token}'")
        
        if next_token == '<END>':
            break
        prev_token = next_token
    
    return output

# Example
input_seq = ['hello', 'world', '<END>']
vocab = ['bonjour', 'monde', '<END>']

print("=== Translation Example ===")
context = simple_encoder(input_seq, hidden_size=256)
output = simple_decoder(context, max_length=5, vocab=vocab)
```

## Teacher Forcing

```python
print("\n=== TEACHER FORCING ===")
print("""
Training strategy for decoder:

WITHOUT teacher forcing:
  - Use model's own predictions as next input
  - Errors compound (exposure bias)
  - Slow training

WITH teacher forcing:
  - Use ground truth as next input
  - Faster convergence
  - Gap between training and inference

Training:    y1_true → y2_true → y3_true (teacher forcing)
Inference:   y1_pred → y2_pred → y3_pred (autoregressive)

SCHEDULED SAMPLING:
  - Gradually reduce teacher forcing
  - Probability p of using ground truth
  - p decreases during training
  - Bridges train/inference gap
""")

def teacher_forcing_example():
    """Demonstrate teacher forcing vs autoregressive"""
    target = ['je', 'suis', 'content', '<END>']
    
    print("Teacher forcing (training):")
    print("  Input to decoder at each step:")
    print(f"  Step 0: '<START>' → predict 'je'")
    print(f"  Step 1: 'je' (true) → predict 'suis'")
    print(f"  Step 2: 'suis' (true) → predict 'content'")
    
    print("\nAutoregressive (inference):")
    print("  Input to decoder at each step:")
    print(f"  Step 0: '<START>' → predict 'je'")
    print(f"  Step 1: 'je' (predicted) → predict 'suis'")
    print(f"  Step 2: 'suis' (predicted) → predict 'content'")

teacher_forcing_example()
```

## Encoder-Decoder in Keras

```python
print("\n=== KERAS IMPLEMENTATION ===")
print("""
# Define encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Define decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, 
                                      initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Training
model.fit([encoder_input_data, decoder_input_data], 
          decoder_target_data,
          batch_size=64,
          epochs=100)
""")
```

## Inference Mode

```python
print("\n=== INFERENCE MODE ===")
print("""
Training and inference use different procedures:

TRAINING:
  - Feed entire input sequence at once
  - Teacher forcing with target sequence
  - Single forward pass

INFERENCE:
  - Encode input once
  - Decode step by step
  - Use previous output as next input
  - Stop at <END> token or max length
""")

print("""
Inference code:

# Encode input
states = encoder_model.predict(input_seq)

# Generate output
target_seq = np.zeros((1, 1, num_decoder_tokens))
target_seq[0, 0, start_token_idx] = 1

output_tokens = []
stop = False

while not stop:
    # Predict next token
    output, h, c = decoder_model.predict([target_seq] + states)
    
    # Sample token
    token_idx = np.argmax(output[0, -1, :])
    output_tokens.append(token_idx)
    
    # Check stop condition
    if token_idx == end_token_idx or len(output_tokens) > max_length:
        stop = True
    
    # Update for next step
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, token_idx] = 1
    states = [h, c]
""")
```

## Beam Search

```python
print("\n=== BEAM SEARCH ===")
print("""
GREEDY DECODING:
  - Always pick most likely token
  - Fast but suboptimal
  - Can miss better sequences

BEAM SEARCH:
  - Keep top K candidates at each step
  - K = beam width
  - Score = sum of log probabilities
  - Returns K-best sequences

Example (beam width = 2):

Step 1: P('I')=0.6, P('We')=0.3, P('The')=0.1
  Keep: ['I'], ['We']

Step 2 for 'I': P('am')=0.5, P('like')=0.4
  Candidates: ['I', 'am'], ['I', 'like']

Step 2 for 'We': P('are')=0.6, P('like')=0.3
  Candidates: ['We', 'are'], ['We', 'like']

Keep top 2 overall by total score.
""")

def beam_search_demo(beam_width=2):
    """Simplified beam search demonstration"""
    # Simulated probabilities
    step1_probs = {'I': 0.6, 'We': 0.3}
    step2_probs = {
        'I': {'am': 0.5, 'like': 0.4},
        'We': {'are': 0.6, 'like': 0.3}
    }
    
    # Step 1: Initialize beams
    beams = [(np.log(p), [token]) for token, p in step1_probs.items()]
    beams = sorted(beams, reverse=True)[:beam_width]
    
    print(f"After step 1 (beam width={beam_width}):")
    for score, seq in beams:
        print(f"  {seq}, score={score:.3f}")
    
    # Step 2: Expand beams
    new_beams = []
    for score, seq in beams:
        last_token = seq[-1]
        for next_token, prob in step2_probs[last_token].items():
            new_score = score + np.log(prob)
            new_seq = seq + [next_token]
            new_beams.append((new_score, new_seq))
    
    # Keep top K
    new_beams = sorted(new_beams, reverse=True)[:beam_width]
    
    print(f"\nAfter step 2:")
    for score, seq in new_beams:
        print(f"  {seq}, score={score:.3f}")

beam_search_demo()
```

## Sequence Generation Strategies

```python
print("\n=== GENERATION STRATEGIES ===")
print("""
1. GREEDY:
   - Always pick argmax
   - Deterministic
   - Can be repetitive

2. BEAM SEARCH:
   - Explore multiple paths
   - Better quality
   - Still deterministic

3. SAMPLING:
   - Sample from probability distribution
   - More diverse
   - Can be incoherent

4. TOP-K SAMPLING:
   - Sample from top K tokens only
   - Balanced diversity/quality
   - K typically 40-100

5. TOP-P (NUCLEUS) SAMPLING:
   - Sample from smallest set with cumulative prob > p
   - Dynamic vocabulary size
   - p typically 0.9-0.95

6. TEMPERATURE:
   - Adjust distribution sharpness
   - T < 1: More confident, less diverse
   - T > 1: More uniform, more diverse
""")

def temperature_sampling(logits, temperature=1.0):
    """Apply temperature to logits before softmax"""
    scaled = logits / temperature
    probs = np.exp(scaled) / np.sum(np.exp(scaled))
    return probs

logits = np.array([2.0, 1.0, 0.5])
print("Effect of temperature:")
for temp in [0.5, 1.0, 2.0]:
    probs = temperature_sampling(logits, temp)
    print(f"  T={temp}: {probs.round(3)}")
```

## Key Points

- **Encoder-Decoder**: Encode input → context → decode output
- **Context bottleneck**: Fixed-size limits long sequences
- **Teacher forcing**: Use ground truth during training
- **Beam search**: Explore multiple candidate sequences
- **Sampling strategies**: Control diversity vs quality
- **Temperature**: Adjust prediction confidence

## Reflection Questions

1. Why is the fixed-size context vector a limitation for long sequences?
2. What is the exposure bias problem with teacher forcing?
3. When would you prefer beam search over sampling?
