# Image Classification with CNNs

## Introduction

Image classification is the task of assigning a label to an image from a set of predefined categories. CNNs have revolutionized this field, achieving superhuman performance on many benchmarks.

## Classification Pipeline

```python
import numpy as np
import pandas as pd

print("=== IMAGE CLASSIFICATION PIPELINE ===")
print("""
Complete pipeline:

1. DATA PREPARATION
   - Collect labeled images
   - Split train/val/test
   - Data augmentation
   - Normalization

2. MODEL SELECTION
   - Architecture choice
   - Pre-trained vs from scratch

3. TRAINING
   - Loss: Cross-entropy
   - Optimizer: Adam, SGD
   - Learning rate schedule
   - Regularization

4. EVALUATION
   - Accuracy, F1-score
   - Confusion matrix
   - Per-class metrics

5. DEPLOYMENT
   - Model export
   - Inference optimization
""")
```

## Data Augmentation

```python
print("\n=== DATA AUGMENTATION ===")
print("""
Artificially expand training data with transformations:

GEOMETRIC:
  - Random crop
  - Horizontal flip
  - Rotation (small angles)
  - Scaling/zoom
  - Translation

COLOR:
  - Brightness adjustment
  - Contrast adjustment
  - Saturation changes
  - Color jittering

ADVANCED:
  - Cutout: Random rectangular masks
  - Mixup: Blend two images and labels
  - CutMix: Replace patches between images
  - AutoAugment: Learned augmentation policies
""")

print("""
Keras ImageDataGenerator example:

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale!

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
""")
```

## Building the Model

```python
print("\n=== BUILDING THE MODEL ===")
print("""
From scratch (small dataset):

model = Sequential([
    # Block 1
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Conv2D(32, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
""")

# Parameter calculation
print("\nModel summary (conceptual):")
layers_info = [
    ("Input", "224×224×3", "-"),
    ("Conv+BN+Conv+BN+Pool", "112×112×32", "~10K"),
    ("Conv+BN+Conv+BN+Pool", "56×56×64", "~75K"),
    ("Conv+BN+Conv+BN+Pool", "28×28×128", "~300K"),
    ("Flatten+Dense+BN", "512", "~50M"),
    ("Dense", "num_classes", "~5K")
]

print(f"{'Layer':<30} {'Output Shape':<15} {'Params':<10}")
print("-" * 55)
for name, shape, params in layers_info:
    print(f"{name:<30} {shape:<15} {params:<10}")
```

## Transfer Learning Approach

```python
print("\n=== TRANSFER LEARNING ===")
print("""
Much better for most cases:

# Load pre-trained model
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base
base_model.trainable = False

# Add custom classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train classifier only
history = model.fit(train_gen, epochs=10, validation_data=val_gen)

# Fine-tune (optional)
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Much smaller LR!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(train_gen, epochs=10, validation_data=val_gen)
""")
```

## Training Strategy

```python
print("\n=== TRAINING STRATEGY ===")
print("""
Recommended approach:

1. START with transfer learning
   - Freeze base model
   - Train only classifier
   - Higher learning rate (1e-3)
   
2. FINE-TUNE if needed
   - Unfreeze later layers
   - Very small learning rate (1e-5)
   - Continue training

3. CALLBACKS:
   - EarlyStopping: Prevent overfitting
   - ReduceLROnPlateau: Adaptive LR
   - ModelCheckpoint: Save best model

4. LEARNING RATE SCHEDULE:
   - Warmup: Start small, increase
   - Cosine annealing: Smooth decay
   - Step decay: Reduce at milestones
""")

print("""
Callbacks example:

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]
""")
```

## Evaluation Metrics

```python
print("\n=== EVALUATION ===")
print("""
Classification metrics:

ACCURACY: Overall correct predictions
  accuracy = correct / total

CONFUSION MATRIX: Predictions vs actual for each class
  - Diagonal: Correct
  - Off-diagonal: Errors

PER-CLASS METRICS:
  Precision = TP / (TP + FP)  # How many predicted positives are correct
  Recall = TP / (TP + FN)     # How many actual positives were found
  F1 = 2 × (P × R) / (P + R)  # Balance of precision/recall

TOP-K ACCURACY:
  - Correct if true label in top K predictions
  - Top-5 accuracy common for many-class problems
""")

# Simulate evaluation
np.random.seed(42)
y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])

print("Example predictions:")
print(f"  True: {y_true}")
print(f"  Pred: {y_pred}")

# Confusion matrix
from collections import Counter

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = confusion_matrix(y_true, y_pred, 3)
print("\nConfusion Matrix:")
print(cm)

accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\nAccuracy: {accuracy:.2%}")
```

## Common Mistakes

```python
print("\n=== COMMON MISTAKES ===")
print("""
1. NOT USING DATA AUGMENTATION
   - Small datasets need it desperately
   - But don't augment validation data!

2. WRONG PREPROCESSING
   - Must match model's expected input
   - ImageNet models: Specific normalization
   - EfficientNet: [0,1] range or specific

3. TOO HIGH LEARNING RATE FOR FINE-TUNING
   - Use 10-100x smaller than initial training
   - Large LR destroys pre-trained features

4. FORGETTING TO FREEZE
   - Initial training: Freeze base model
   - Unfrozen model + high LR = disaster

5. NOT MONITORING VALIDATION
   - Training accuracy alone is meaningless
   - Watch for train/val gap

6. WRONG DATA SPLIT
   - Information leakage between splits
   - Similar images in train and test
""")
```

## Key Points

- **Data augmentation**: Essential for small datasets, apply to training only
- **Transfer learning**: Use pre-trained models, train classifier first
- **Fine-tuning**: Small learning rate, freeze early layers
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Evaluation**: Confusion matrix, per-class metrics, not just accuracy
- **Preprocessing**: Must match model's training preprocessing

## Reflection Questions

1. Why is data augmentation applied only to training data and not validation?
2. What happens if you fine-tune with the same learning rate used for training the classifier?
3. How would you diagnose whether your model is overfitting vs underfitting?
