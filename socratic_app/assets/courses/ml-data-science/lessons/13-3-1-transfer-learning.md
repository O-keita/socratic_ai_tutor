# Transfer Learning with Transformers

## Introduction

Transfer learning with transformers leverages pre-trained models to achieve strong performance on downstream tasks with minimal task-specific training.

## Pre-training and Fine-tuning Paradigm

```python
import numpy as np
import pandas as pd

print("=== PRE-TRAIN AND FINE-TUNE ===")
print("""
TWO-STAGE APPROACH:

1. PRE-TRAINING (expensive, done once):
   - Train on massive unlabeled data
   - Learn general language/vision features
   - Billions of tokens/images
   - Weeks on many GPUs

2. FINE-TUNING (cheap, per task):
   - Start from pre-trained weights
   - Train on small labeled dataset
   - Learn task-specific patterns
   - Hours on single GPU

Benefits:
  - Most learning already done
  - Works with small datasets
  - Faster convergence
  - Better generalization
""")

print("""
Analogy:
  Pre-training = Learning to read/write
  Fine-tuning = Learning a specific job
  
  You don't re-learn language for each new job!
""")
```

## Types of Transfer

```python
print("\n=== TRANSFER LEARNING APPROACHES ===")
print("""
1. FEATURE EXTRACTION
   - Freeze pre-trained weights
   - Add new classifier head
   - Only train the head
   
   Best when:
     - Very small dataset
     - Task similar to pre-training
     - Want fast training

2. FINE-TUNING (full)
   - Unfreeze all weights
   - Train entire model
   - Use small learning rate
   
   Best when:
     - Medium-sized dataset
     - Task differs from pre-training
     - Have compute resources

3. PARTIAL FINE-TUNING
   - Freeze early layers
   - Fine-tune later layers + head
   - Balance between above
   
   Best when:
     - Domain differs moderately
     - Want stability + adaptability
""")
```

## Using Hugging Face Transformers

```python
print("\n=== HUGGING FACE LIBRARY ===")
print("""
The standard library for transformer models:

# Installation
pip install transformers datasets

# Load pre-trained model
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# For classification
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Number of classes
)

Available model types:
  - AutoModelForSequenceClassification
  - AutoModelForTokenClassification (NER)
  - AutoModelForQuestionAnswering
  - AutoModelForCausalLM (text generation)
  - AutoModelForMaskedLM
  - And many more...
""")
```

## Fine-tuning BERT for Classification

```python
print("\n=== FINE-TUNING EXAMPLE ===")
print("""
# Complete fine-tuning pipeline

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 1. Load dataset
dataset = load_dataset('imdb')

# 2. Load tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 3. Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,  # Small LR for fine-tuning!
    evaluation_strategy='epoch',
)

# 5. Create Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()
""")
```

## Learning Rate and Scheduling

```python
print("\n=== FINE-TUNING BEST PRACTICES ===")
print("""
LEARNING RATE:
  Pre-training: 1e-4
  Fine-tuning: 2e-5 to 5e-5 (10-20× smaller!)
  
  Why smaller?
    - Pre-trained weights are good
    - Don't want to destroy them
    - Small adjustments sufficient

WARMUP:
  - Start with very small LR
  - Linearly increase to target
  - Stabilizes early training
  
  warmup_steps = 500  # or 10% of training

WEIGHT DECAY:
  - L2 regularization
  - Typically 0.01
  - Prevents overfitting

EPOCHS:
  - Usually 2-4 epochs sufficient
  - More can overfit to small data
""")

print("""
Learning rate schedules:

Constant:     ────────────────
              
Linear decay: ───╲
                  ╲──────────
                  
Cosine:       ───╲___/╲___
                     (cycles)
                     
Warmup + linear:
              ╱────╲
             ╱      ╲──────
""")
```

## Domain Adaptation

```python
print("\n=== DOMAIN ADAPTATION ===")
print("""
When target domain differs from pre-training:

STRATEGIES:

1. CONTINUED PRE-TRAINING
   - Further pre-train on domain data (unlabeled)
   - Then fine-tune on task (labeled)
   
   Example: BERT → BioBERT (biomedical) → Disease classification

2. GRADUAL UNFREEZING
   - Start with top layers only
   - Progressively unfreeze lower layers
   - Prevents catastrophic forgetting
   
   Epoch 1: Train only classifier head
   Epoch 2: Unfreeze last transformer layer
   Epoch 3: Unfreeze more layers
   ...

3. DISCRIMINATIVE FINE-TUNING
   - Different learning rates per layer
   - Lower layers: smaller LR
   - Higher layers: larger LR
   
   LR = base_lr * (decay_factor ^ layer_index)
""")

def layer_wise_lr(model, base_lr=2e-5, decay=0.95, num_layers=12):
    """Calculate layer-wise learning rates"""
    lr_dict = {}
    
    # Embedding layers - smallest LR
    lr_dict['embeddings'] = base_lr * (decay ** (num_layers + 1))
    
    # Transformer layers
    for layer in range(num_layers):
        lr_dict[f'layer_{layer}'] = base_lr * (decay ** (num_layers - layer))
    
    # Classifier head - base LR
    lr_dict['classifier'] = base_lr
    
    print("Layer-wise learning rates:")
    for name, lr in list(lr_dict.items())[:5]:
        print(f"  {name}: {lr:.2e}")
    print("  ...")
    print(f"  classifier: {lr_dict['classifier']:.2e}")

layer_wise_lr(None)
```

## Zero-Shot and Few-Shot

```python
print("\n=== ZERO-SHOT / FEW-SHOT ===")
print("""
Without fine-tuning:

ZERO-SHOT CLASSIFICATION:
  Use model as-is with clever prompting
  
  # Using Hugging Face pipeline
  from transformers import pipeline
  
  classifier = pipeline('zero-shot-classification',
                        model='facebook/bart-large-mnli')
  
  result = classifier(
      "This movie was absolutely fantastic!",
      candidate_labels=['positive', 'negative', 'neutral']
  )
  # {'labels': ['positive', 'negative', 'neutral'],
  #  'scores': [0.95, 0.03, 0.02]}

FEW-SHOT with SetFit:
  - Fine-tune with just 8-16 examples per class
  - Uses sentence transformers + contrastive learning
  
  from setfit import SetFitModel
  
  model = SetFitModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
  model.fit(train_dataset)  # Very few examples!
""")
```

## Model Selection

```python
print("\n=== CHOOSING A PRE-TRAINED MODEL ===")
print("""
Factors to consider:

1. TASK TYPE
   - Classification: BERT, RoBERTa, DeBERTa
   - Generation: GPT-2, T5, BART
   - Embeddings: Sentence-BERT, all-mpnet
   - QA: BERT, RoBERTa, T5

2. DOMAIN
   - General: BERT, RoBERTa
   - Scientific: SciBERT
   - Biomedical: BioBERT, PubMedBERT
   - Legal: Legal-BERT
   - Code: CodeBERT, CodeT5

3. SIZE vs PERFORMANCE
   - Base (~110M): Fast, decent
   - Large (~340M): Better, slower
   - XL/XXL: Best, very slow
   
   Consider your compute budget!

4. LANGUAGE
   - English: Most models
   - Multilingual: mBERT, XLM-R
   - Specific language: Language-specific models

Popular models (2024):
  - RoBERTa: Robust BERT, better pre-training
  - DeBERTa: State-of-the-art for understanding
  - T5: Unified text-to-text format
  - Llama: Open source, good generation
""")

model_comparison = pd.DataFrame({
    'Model': ['BERT-base', 'RoBERTa-base', 'DeBERTa-base', 'T5-base'],
    'Parameters': ['110M', '125M', '139M', '220M'],
    'Best For': ['General', 'Understanding', 'Understanding', 'Seq2Seq'],
    'Speed': ['Fast', 'Fast', 'Medium', 'Medium']
})
print("\nModel comparison:")
print(model_comparison.to_string(index=False))
```

## Key Points

- **Pre-train then fine-tune**: Leverage massive pre-training
- **Small learning rate**: 2e-5 to 5e-5 for fine-tuning
- **Feature extraction**: Freeze and add head for small data
- **Domain adaptation**: Continued pre-training helps
- **Hugging Face**: Standard library for transformers
- **Model selection**: Match model to task and compute

## Reflection Questions

1. Why use a smaller learning rate for fine-tuning than pre-training?
2. When would you choose feature extraction over full fine-tuning?
3. How does domain-specific pre-training improve downstream tasks?
