# Building Neural Networks with Keras

## Introduction

Keras provides a high-level API for building and training neural networks. It abstracts away the complexity while allowing full customization when needed.

## Keras Sequential API

```python
import numpy as np
import pandas as pd

print("=== KERAS SEQUENTIAL API ===")
print("""
Sequential model: Stack of layers in order

from tensorflow import keras
from keras import layers

model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

Sequential is best for:
  - Simple feed-forward networks
  - One input, one output
  - Linear stack of layers
""")

# Simulating Keras-like API
class Dense:
    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
    
    def __repr__(self):
        return f"Dense({self.units}, activation='{self.activation}')"

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers or []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def summary(self):
        print("Model: Sequential")
        print("-" * 50)
        total_params = 0
        prev_units = self.layers[0].input_shape[0] if self.layers[0].input_shape else 0
        
        for i, layer in enumerate(self.layers):
            if i == 0 and layer.input_shape:
                prev_units = layer.input_shape[0]
            
            params = prev_units * layer.units + layer.units  # weights + biases
            total_params += params
            print(f"dense_{i}: units={layer.units}, params={params}")
            prev_units = layer.units
        
        print("-" * 50)
        print(f"Total params: {total_params}")

# Build a model
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

print("Model architecture:")
model.summary()
```

## Functional API

```python
print("\n=== FUNCTIONAL API ===")
print("""
For more complex architectures:
  - Multiple inputs/outputs
  - Shared layers
  - Non-linear connectivity

from keras import Model, Input

# Define inputs
inputs = Input(shape=(784,))

# Build layers
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

Benefits:
  - Full control over connections
  - Easy to create branches, merges
  - Can access intermediate layers
""")

print("""
Example: Multi-input model

# Two inputs
input_a = Input(shape=(32,), name='input_a')
input_b = Input(shape=(64,), name='input_b')

# Process each input
x_a = Dense(16, activation='relu')(input_a)
x_b = Dense(16, activation='relu')(input_b)

# Merge
merged = layers.concatenate([x_a, x_b])

# Output
output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_a, input_b], outputs=output)
""")
```

## Compiling the Model

```python
print("\n=== COMPILING THE MODEL ===")
print("""
Configure training with compile():

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

OPTIMIZER options:
  - 'sgd': Stochastic Gradient Descent
  - 'adam': Adam (most common)
  - 'rmsprop': RMSprop
  - keras.optimizers.Adam(learning_rate=0.001)

LOSS options:
  - 'mse': Mean Squared Error (regression)
  - 'binary_crossentropy': Binary classification
  - 'categorical_crossentropy': Multi-class (one-hot)
  - 'sparse_categorical_crossentropy': Multi-class (integers)

METRICS: Track during training
  - 'accuracy'
  - 'mae'
  - Custom metrics
""")

print("""
Match loss to output:

┌─────────────────────┬────────────────────────────┬─────────────┐
│ Task                │ Output Activation          │ Loss        │
├─────────────────────┼────────────────────────────┼─────────────┤
│ Binary classif.     │ sigmoid (1 unit)           │ binary_ce   │
│ Multi-class         │ softmax (N units)          │ categorical │
│ Regression          │ linear/none                │ mse/mae     │
│ Multi-label         │ sigmoid (N units)          │ binary_ce   │
└─────────────────────┴────────────────────────────┴─────────────┘
""")
```

## Training the Model

```python
print("\n=== TRAINING THE MODEL ===")
print("""
Use fit() to train:

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

Key parameters:
  - epochs: Number of complete passes through data
  - batch_size: Samples per gradient update
  - validation_split: Fraction for validation
  - validation_data: (X_val, y_val) explicitly
  - callbacks: Functions called during training
  - verbose: 0=silent, 1=progress bar, 2=one line/epoch
""")

# Simulate training history
print("Training output:")
print("""
Epoch 1/10
1875/1875 [======] - 5s - loss: 0.4521 - accuracy: 0.8632 - val_loss: 0.2341 - val_accuracy: 0.9234
Epoch 2/10
1875/1875 [======] - 4s - loss: 0.2156 - accuracy: 0.9341 - val_loss: 0.1876 - val_accuracy: 0.9456
...
""")
```

## Common Layers

```python
print("\n=== COMMON LAYERS ===")
print("""
DENSE (fully connected):
  Dense(units, activation='relu')
  
DROPOUT (regularization):
  Dropout(rate=0.5)
  
BATCH NORMALIZATION:
  BatchNormalization()
  
FLATTEN (for CNNs):
  Flatten()
  
EMBEDDING (for text):
  Embedding(vocab_size, embed_dim)
  
CONVOLUTIONAL:
  Conv2D(filters=32, kernel_size=3, activation='relu')
  MaxPooling2D(pool_size=2)
  
RECURRENT:
  LSTM(units=64, return_sequences=True)
  GRU(units=64)
""")

print("""
Example: Complete classification network

model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(10, activation='softmax')
])
""")
```

## Callbacks

```python
print("\n=== CALLBACKS ===")
print("""
Callbacks are hooks into training process:

1. EARLY STOPPING:
   EarlyStopping(
       monitor='val_loss',
       patience=5,
       restore_best_weights=True
   )

2. MODEL CHECKPOINT:
   ModelCheckpoint(
       filepath='best_model.keras',
       monitor='val_accuracy',
       save_best_only=True
   )

3. LEARNING RATE SCHEDULER:
   ReduceLROnPlateau(
       monitor='val_loss',
       factor=0.5,
       patience=3
   )

4. TENSORBOARD:
   TensorBoard(log_dir='./logs')

5. CSV LOGGER:
   CSVLogger('training_log.csv')
""")

print("""
Using callbacks:

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(X_train, y_train, 
                    epochs=100, 
                    callbacks=callbacks)
""")
```

## Evaluating and Predicting

```python
print("\n=== EVALUATION AND PREDICTION ===")
print("""
EVALUATE on test data:
  loss, accuracy = model.evaluate(X_test, y_test)

PREDICT:
  predictions = model.predict(X_new)
  
  # For classification, get class labels:
  predicted_classes = np.argmax(predictions, axis=1)

PREDICT on single sample:
  sample = X_test[0:1]  # Keep batch dimension
  pred = model.predict(sample)
""")

# Simulate predictions
print("Example predictions (softmax output):")
print("""
[0.02, 0.01, 0.03, 0.85, 0.02, 0.01, 0.03, 0.01, 0.01, 0.01]
                     ^
              Predicted class 3 (85% confidence)
""")
```

## Saving and Loading

```python
print("\n=== SAVING AND LOADING ===")
print("""
SAVE entire model:
  model.save('my_model.keras')  # New format
  model.save('my_model.h5')     # Legacy HDF5

LOAD model:
  model = keras.models.load_model('my_model.keras')

SAVE weights only:
  model.save_weights('weights.weights.h5')

LOAD weights:
  model.load_weights('weights.weights.h5')

EXPORT for serving:
  model.export('exported_model')  # SavedModel format
""")
```

## Key Points

- **Sequential API**: Simple stack of layers
- **Functional API**: Complex architectures, multiple I/O
- **compile()**: Set optimizer, loss, metrics
- **fit()**: Train with epochs, batch_size, validation
- **Callbacks**: Early stopping, checkpoints, LR scheduling
- **evaluate()**: Test set performance
- **predict()**: Get model outputs
- **save/load**: Persist models

## Reflection Questions

1. When would you use Functional API instead of Sequential?
2. Why is it important to use validation data during training?
3. How do you choose the right combination of optimizer and learning rate?
