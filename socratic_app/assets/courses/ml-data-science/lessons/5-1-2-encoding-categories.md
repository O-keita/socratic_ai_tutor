# Encoding Categorical Variables

## Introduction

Machine learning algorithms require numerical inputs. Categorical variables must be encoded into numeric representations while preserving their informational content appropriately.

## Types of Categorical Variables

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

np.random.seed(42)

print("=== TYPES OF CATEGORICAL VARIABLES ===")
print("""
1. NOMINAL (No Order)
   - Categories with no inherent order
   - Examples: Color, Country, Product Type
   - Encoding: One-hot encoding preferred

2. ORDINAL (Ordered)
   - Categories with meaningful order
   - Examples: Size (S/M/L), Rating (Poor/Good/Excellent)
   - Encoding: Label encoding or ordinal encoding

3. BINARY
   - Only two categories
   - Examples: Yes/No, Male/Female, True/False
   - Encoding: Simple 0/1 mapping

4. HIGH CARDINALITY
   - Many unique categories
   - Examples: ZIP codes, User IDs
   - Encoding: Target encoding, hash encoding
""")

# Create example data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],      # Nominal
    'size': ['small', 'medium', 'large', 'medium', 'small'], # Ordinal
    'active': ['yes', 'no', 'yes', 'yes', 'no'],            # Binary
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston']       # Nominal (higher cardinality)
})
print("\nSample Data:")
print(df)
```

## Label Encoding

```python
print("\n=== LABEL ENCODING ===")
print("""
Maps each category to an integer.

Warning: Only use for ordinal variables or tree-based models!
Linear models will interpret as numeric relationships.

"red=0, blue=1, green=2" implies red < blue < green
""")

# Label encoding
le = LabelEncoder()
df_le = df.copy()
df_le['color_encoded'] = le.fit_transform(df['color'])

print("Label encoded color:")
print(df_le[['color', 'color_encoded']])
print(f"\nClasses: {le.classes_}")

# Ordinal encoding with proper order
print("\n=== ORDINAL ENCODING ===")
print("For ordered categories, specify the order:")

oe = OrdinalEncoder(categories=[['small', 'medium', 'large']])
df_le['size_encoded'] = oe.fit_transform(df[['size']])

print("Ordinal encoded size:")
print(df_le[['size', 'size_encoded']])
```

## One-Hot Encoding

```python
print("\n=== ONE-HOT ENCODING ===")
print("""
Creates binary columns for each category.

Original: color = ['red', 'blue', 'green']
One-Hot:
  color_red:   [1, 0, 0]
  color_blue:  [0, 1, 0]
  color_green: [0, 0, 1]

Advantages:
  - No ordinal relationship implied
  - Works with any ML algorithm
  
Disadvantages:
  - Increases dimensionality
  - Can cause multicollinearity
""")

# Using pandas get_dummies
print("Using pandas get_dummies:")
df_onehot = pd.get_dummies(df['color'], prefix='color')
print(df_onehot)

# Using sklearn OneHotEncoder
print("\nUsing sklearn OneHotEncoder:")
ohe = OneHotEncoder(sparse_output=False)
color_encoded = ohe.fit_transform(df[['color']])
feature_names = ohe.get_feature_names_out(['color'])
df_ohe = pd.DataFrame(color_encoded, columns=feature_names)
print(df_ohe)

# Drop first to avoid multicollinearity
print("\nDrop first (avoid dummy variable trap):")
df_onehot_drop = pd.get_dummies(df['color'], prefix='color', drop_first=True)
print(df_onehot_drop)
```

## Binary Encoding

```python
print("\n=== BINARY ENCODING ===")
print("""
Encodes categories as binary numbers.

Example with 8 categories (requires 3 binary columns):
  cat_0: 000
  cat_1: 001
  cat_2: 010
  ...
  cat_7: 111

Advantages:
  - Fewer columns than one-hot
  - Good for high cardinality
  
Disadvantages:
  - Proximity in binary space may not be meaningful
""")

# Manual binary encoding example
categories = ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5', 'cat_6', 'cat_7']
n_bits = int(np.ceil(np.log2(len(categories))))
print(f"\n{len(categories)} categories need {n_bits} binary columns:")

for i, cat in enumerate(categories):
    binary = format(i, f'0{n_bits}b')
    print(f"  {cat}: {binary}")
```

## Target Encoding

```python
print("\n=== TARGET ENCODING ===")
print("""
Replaces category with mean of target variable for that category.

Advantages:
  - Single column regardless of cardinality
  - Captures relationship with target
  
Disadvantages:
  - Risk of overfitting
  - Data leakage if not done carefully
  - Requires target variable
  
Must use cross-validation to prevent leakage!
""")

# Sample data with target
df_target = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'target': [100, 120, 200, 180, 150, 170, 110, 220, 140, 130]
})

print("Original data:")
print(df_target)

# Calculate target means per category
target_means = df_target.groupby('category')['target'].mean()
print("\nTarget means by category:")
print(target_means)

# Apply encoding
df_target['category_encoded'] = df_target['category'].map(target_means)
print("\nAfter target encoding:")
print(df_target)

print("""
To prevent leakage:
  - Use leave-one-out encoding
  - Use cross-validation encoding
  - Add smoothing/regularization
""")
```

## Frequency Encoding

```python
print("\n=== FREQUENCY ENCODING ===")
print("""
Replaces category with its frequency (count or proportion).

Advantages:
  - Simple to implement
  - No target leakage
  - Good for tree-based models
  
Disadvantages:
  - Categories with same frequency get same value
  - Loses category identity
""")

df_freq = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B', 'C', 'D', 'D', 'D', 'D']
})

print("Category counts:")
freq = df_freq['category'].value_counts()
print(freq)

# Frequency encoding (count)
df_freq['freq_count'] = df_freq['category'].map(freq)

# Frequency encoding (proportion)
df_freq['freq_prop'] = df_freq['category'].map(freq / len(df_freq))

print("\nFrequency encoded:")
print(df_freq)
```

## Handling High Cardinality

```python
print("\n=== HIGH CARDINALITY STRATEGIES ===")
print("""
When categories are numerous (e.g., ZIP codes, user IDs):

1. GROUP RARE CATEGORIES
   - Combine low-frequency categories into "Other"
   
2. TARGET ENCODING
   - Encode with target mean
   
3. FEATURE HASHING
   - Hash categories to fixed number of features
   
4. EMBEDDING
   - Learn dense representations (neural networks)
   
5. AGGREGATION
   - Use higher-level grouping (ZIP → State)
""")

# Example: Group rare categories
np.random.seed(42)
df_rare = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 1000, 
                                 p=[0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02])
})

print("Original category distribution:")
print(df_rare['category'].value_counts())

# Group categories with <5% frequency into "Other"
threshold = 0.05
freq = df_rare['category'].value_counts(normalize=True)
rare_categories = freq[freq < threshold].index.tolist()
print(f"\nRare categories (<5%): {rare_categories}")

df_rare['category_grouped'] = df_rare['category'].apply(
    lambda x: 'Other' if x in rare_categories else x
)

print("\nAfter grouping:")
print(df_rare['category_grouped'].value_counts())
```

## Encoding Selection Guide

```python
print("\n=== ENCODING SELECTION GUIDE ===")
print("""
NOMINAL CATEGORIES:
  ✓ One-hot encoding (low cardinality)
  ✓ Target encoding (high cardinality, with caution)
  ✓ Frequency encoding (tree models)

ORDINAL CATEGORIES:
  ✓ Ordinal encoding (preserve order)
  ✓ Label encoding

BINARY CATEGORIES:
  ✓ Simple 0/1 mapping
  ✓ One column is sufficient

HIGH CARDINALITY:
  ✓ Target encoding (supervised)
  ✓ Frequency encoding
  ✓ Group rare categories
  ✓ Feature hashing
  
TREE-BASED MODELS:
  ✓ Label encoding often works
  ✓ One-hot can degrade performance
  
LINEAR MODELS / NEURAL NETWORKS:
  ✓ One-hot encoding preferred
  ✓ Target encoding with regularization

IMPORTANT: Handle unknown categories!
  - What happens with new categories in test data?
  - Use handle_unknown='ignore' in sklearn
""")
```

## Key Points

- **Label encoding**: Simple but implies ordering; use for ordinal or trees
- **One-hot encoding**: Safe for nominal; increases dimensionality
- **Ordinal encoding**: Preserves meaningful order
- **Target encoding**: Powerful but risk of leakage
- **Frequency encoding**: Simple, no leakage
- **High cardinality**: Group rare categories or use specialized methods
- **Handle unknowns**: Plan for new categories in production

## Reflection Questions

1. Why is one-hot encoding preferred for linear models with nominal variables?
2. How can target encoding lead to data leakage?
3. When would you choose frequency encoding over one-hot encoding?
