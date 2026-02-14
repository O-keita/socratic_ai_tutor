# Data Cleaning Strategies

## Introduction

Data cleaning is the process of detecting and correcting errors, inconsistencies, and inaccuracies in datasets. Clean data is essential for reliable analysis and modeling.

## Common Data Quality Issues

```python
import numpy as np
import pandas as pd

np.random.seed(42)

print("=== DATA QUALITY ISSUES ===")
print("""
1. MISSING VALUES
   - Empty cells, NaN, NULL
   - Placeholder values (-999, "N/A", "Unknown")

2. DUPLICATE RECORDS
   - Exact duplicates
   - Partial duplicates (same entity, different formats)

3. INCONSISTENT FORMATS
   - Dates: "2023-01-15", "01/15/2023", "Jan 15, 2023"
   - Names: "John Smith", "JOHN SMITH", "Smith, John"
   - Categories: "USA", "U.S.A.", "United States"

4. INVALID VALUES
   - Negative ages
   - Percentages > 100
   - Future dates for past events

5. OUTLIERS AND ERRORS
   - Typos (1000 instead of 100)
   - Unit confusion (kg vs lbs)

6. STRUCTURAL ISSUES
   - Merged cells
   - Multiple values in single field
   - Inconsistent column names
""")
```

## Detecting Duplicates

```python
print("\n=== DETECTING DUPLICATES ===")

# Create data with duplicates
df = pd.DataFrame({
    'id': [1, 2, 3, 2, 4, 5, 3],
    'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Eve', 'Charlie'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com',
              'bob@email.com', 'david@email.com', 'eve@email.com', 'charlie@test.com'],
    'amount': [100, 200, 300, 200, 400, 500, 350]
})
print("Original Data:")
print(df)

# Check for exact duplicates
print(f"\nExact duplicates: {df.duplicated().sum()}")
print(df[df.duplicated(keep=False)])

# Check for duplicates based on specific columns
print(f"\nDuplicates by 'id': {df.duplicated(subset=['id']).sum()}")
print(df[df.duplicated(subset=['id'], keep=False)])

# Check for duplicates by 'name'
print(f"\nDuplicates by 'name': {df.duplicated(subset=['name']).sum()}")
print(df[df.duplicated(subset=['name'], keep=False)])

# Remove duplicates
df_dedup = df.drop_duplicates()
print(f"\nAfter dropping exact duplicates: {len(df_dedup)} rows")

# Remove duplicates, keep last occurrence
df_dedup_last = df.drop_duplicates(subset=['id'], keep='last')
print(f"After dropping by 'id' (keep last): {len(df_dedup_last)} rows")
print(df_dedup_last)
```

## String Cleaning

```python
print("\n=== STRING CLEANING ===")

# Create messy string data
df = pd.DataFrame({
    'name': ['  John Smith  ', 'JANE DOE', 'bob jones', 'Alice   Brown'],
    'city': ['new york', 'LOS ANGELES', 'Chicago  ', '  boston'],
    'country': ['USA', 'U.S.A.', 'United States', 'US']
})
print("Messy Data:")
print(df)

# Strip whitespace
df['name'] = df['name'].str.strip()
df['city'] = df['city'].str.strip()

# Standardize case
df['name'] = df['name'].str.title()
df['city'] = df['city'].str.title()

print("\nAfter stripping and title case:")
print(df)

# Standardize categorical values
country_mapping = {
    'USA': 'United States',
    'U.S.A.': 'United States',
    'US': 'United States'
}
df['country'] = df['country'].replace(country_mapping)
print("\nAfter standardizing countries:")
print(df)

# Remove special characters
messy_strings = pd.Series(['Hello!@#', 'World$%^', '123&*('])
cleaned = messy_strings.str.replace(r'[^\w\s]', '', regex=True)
print(f"\nRemoving special characters:")
print(f"  Before: {messy_strings.tolist()}")
print(f"  After: {cleaned.tolist()}")
```

## Date Parsing and Standardization

```python
print("\n=== DATE CLEANING ===")

# Various date formats
dates = pd.Series([
    '2023-01-15',
    '01/15/2023',
    'January 15, 2023',
    '15-Jan-2023',
    '2023/01/15',
    '20230115'
])
print("Various date formats:")
print(dates)

# Parse dates
parsed = pd.to_datetime(dates, format='mixed')
print("\nParsed dates:")
print(parsed)

# Standardize to ISO format
standardized = parsed.dt.strftime('%Y-%m-%d')
print("\nStandardized (ISO format):")
print(standardized)

# Handle invalid dates
dates_with_errors = pd.Series(['2023-01-15', 'invalid', '2023-02-30', '2023-13-01'])
print("\nHandling invalid dates:")
parsed_safe = pd.to_datetime(dates_with_errors, errors='coerce')
print(parsed_safe)
print(f"Invalid dates: {parsed_safe.isna().sum()}")
```

## Type Conversion

```python
print("\n=== TYPE CONVERSION ===")

# Data with incorrect types
df = pd.DataFrame({
    'id': ['1', '2', '3', '4'],
    'amount': ['100.50', '200.00', '300.75', 'N/A'],
    'is_active': ['true', 'false', 'True', 'FALSE'],
    'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']
})
print("Original types:")
print(df.dtypes)
print(df)

# Convert numeric (handle errors)
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
print(f"\nAmount converted (with coerce): {df['amount'].tolist()}")

# Convert boolean
bool_mapping = {'true': True, 'false': False}
df['is_active'] = df['is_active'].str.lower().map(bool_mapping)
print(f"Boolean converted: {df['is_active'].tolist()}")

# Convert dates
df['date'] = pd.to_datetime(df['date'])
print(f"Date converted: {df['date'].dtype}")

# Convert ID to int
df['id'] = df['id'].astype(int)

print("\nFinal types:")
print(df.dtypes)
```

## Handling Inconsistent Categories

```python
print("\n=== CATEGORY STANDARDIZATION ===")

# Messy categorical data
categories = pd.Series([
    'Small', 'small', 'SMALL', 'S', 'sm',
    'Medium', 'MEDIUM', 'M', 'med',
    'Large', 'large', 'L', 'lg', 'LG'
])
print("Messy categories:")
print(categories.value_counts())

# Create mapping
category_mapping = {
    'small': 'Small', 'SMALL': 'Small', 's': 'Small', 'sm': 'Small',
    'medium': 'Medium', 'MEDIUM': 'Medium', 'm': 'Medium', 'med': 'Medium',
    'large': 'Large', 'LARGE': 'Large', 'l': 'Large', 'lg': 'Large', 'LG': 'Large'
}
cleaned = categories.str.lower().map(lambda x: category_mapping.get(x, x))
print("\nCleaned categories:")
print(cleaned.value_counts())

# Using fuzzy matching for approximate matches
from difflib import get_close_matches

def standardize_category(value, valid_options, cutoff=0.6):
    """Find closest match to valid option."""
    matches = get_close_matches(value.lower(), [v.lower() for v in valid_options], n=1, cutoff=cutoff)
    if matches:
        idx = [v.lower() for v in valid_options].index(matches[0])
        return valid_options[idx]
    return value

valid = ['Small', 'Medium', 'Large']
test_values = ['smll', 'medm', 'lrge']
for val in test_values:
    print(f"  '{val}' → '{standardize_category(val, valid)}'")
```

## Data Validation

```python
print("\n=== DATA VALIDATION ===")

# Create sample data
df = pd.DataFrame({
    'age': [25, 150, -5, 35, 45],  # Invalid: 150, -5
    'email': ['valid@email.com', 'invalid-email', 'test@test.com', '', 'user@domain.org'],
    'percentage': [50, 101, 75, -10, 95],  # Invalid: 101, -10
    'date': pd.to_datetime(['2023-01-01', '2025-12-01', '2022-06-15', '2023-03-01', '2021-01-01'])
})

print("Data to validate:")
print(df)

# Validation rules
print("\n=== VALIDATION RESULTS ===")

# Age validation (0-120)
invalid_age = (df['age'] < 0) | (df['age'] > 120)
print(f"Invalid ages: {df.loc[invalid_age, 'age'].tolist()}")

# Percentage validation (0-100)
invalid_pct = (df['percentage'] < 0) | (df['percentage'] > 100)
print(f"Invalid percentages: {df.loc[invalid_pct, 'percentage'].tolist()}")

# Email validation (simple regex)
import re
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
invalid_email = ~df['email'].str.match(email_pattern, na=False)
print(f"Invalid emails: {df.loc[invalid_email, 'email'].tolist()}")

# Date validation (not in future)
import datetime
invalid_date = df['date'] > datetime.datetime.now()
print(f"Future dates: {df.loc[invalid_date, 'date'].tolist()}")

# Summary
print("\nValidation Summary:")
print(f"  Total rows: {len(df)}")
print(f"  Rows with any issue: {(invalid_age | invalid_pct | invalid_email | invalid_date).sum()}")
```

## Cleaning Pipeline

```python
print("\n=== CLEANING PIPELINE ===")

def clean_dataframe(df):
    """Apply standard cleaning operations."""
    df = df.copy()
    
    # 1. Remove exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"1. Removed {n_before - len(df)} duplicate rows")
    
    # 2. Strip whitespace from string columns
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    print(f"2. Stripped whitespace from {len(string_cols)} string columns")
    
    # 3. Standardize case for string columns
    for col in string_cols:
        df[col] = df[col].str.title()
    print(f"3. Standardized case")
    
    # 4. Replace empty strings with NaN
    df = df.replace('', np.nan)
    print(f"4. Replaced empty strings with NaN")
    
    # 5. Report missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"5. Missing values:\n{missing[missing > 0]}")
    
    return df

# Example usage
sample_df = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'charlie', 'Alice'],
    'city': ['New York', '  boston  ', '', 'new york'],
    'value': [100, 200, 300, 100]
})
print("\nOriginal:")
print(sample_df)

print("\nCleaning...")
cleaned_df = clean_dataframe(sample_df)

print("\nCleaned:")
print(cleaned_df)
```

## Key Points

- **Duplicates**: Check for exact and partial duplicates
- **String cleaning**: Strip, standardize case, remove special characters
- **Date parsing**: Use pd.to_datetime with error handling
- **Type conversion**: Use errors='coerce' for graceful handling
- **Validation**: Define rules, check data, report violations
- **Document everything**: Keep track of cleaning steps and decisions
- **Automate**: Create reusable cleaning pipelines

## Reflection Questions

1. How would you decide whether to fix or remove an invalid data point?
2. What are the risks of over-cleaning data?
3. How would you validate that your cleaning steps haven't introduced errors?
