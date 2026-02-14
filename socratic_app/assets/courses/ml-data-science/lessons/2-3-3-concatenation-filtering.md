# Concatenation and Filtering Joins

## Introduction

Concatenation combines DataFrames by stacking them vertically or horizontally. Filtering joins (semi-joins and anti-joins) filter one table based on the presence or absence of matches in another table.

## Vertical Concatenation (Stacking Rows)

```python
import pandas as pd
import numpy as np

# Sample data: quarterly sales
q1_sales = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'sales': [100, 150, 200],
    'quarter': 'Q1'
})

q2_sales = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'sales': [120, 140, 220],
    'quarter': 'Q2'
})

q3_sales = pd.DataFrame({
    'product': ['A', 'B', 'D'],  # Note: D instead of C
    'sales': [130, 160, 180],
    'quarter': 'Q3'
})

print("=== VERTICAL CONCATENATION ===")
print(f"Q1:\n{q1_sales}\n")
print(f"Q2:\n{q2_sales}\n")
print(f"Q3:\n{q3_sales}\n")

# Concatenate vertically (stack rows)
all_sales = pd.concat([q1_sales, q2_sales, q3_sales], ignore_index=True)
print(f"Concatenated (all quarters):\n{all_sales}")
```

## Concatenation Options

```python
print("\n=== CONCATENATION OPTIONS ===")

# Without ignore_index (keeps original indices)
concat_with_index = pd.concat([q1_sales, q2_sales])
print(f"Without ignore_index:\n{concat_with_index}\n")

# With ignore_index (resets index)
concat_reset = pd.concat([q1_sales, q2_sales], ignore_index=True)
print(f"With ignore_index:\n{concat_reset}\n")

# Add keys to identify source
concat_keys = pd.concat([q1_sales, q2_sales], keys=['Q1', 'Q2'])
print(f"With keys:\n{concat_keys}")
```

## Handling Different Columns

```python
print("\n=== DIFFERENT COLUMNS ===")

df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'B': [5, 6],
    'C': [7, 8]
})

print(f"df1:\n{df1}\n")
print(f"df2:\n{df2}\n")

# Default: outer join (keep all columns, fill with NaN)
concat_outer = pd.concat([df1, df2], ignore_index=True)
print(f"Outer join (default):\n{concat_outer}\n")

# Inner join (keep only common columns)
concat_inner = pd.concat([df1, df2], join='inner', ignore_index=True)
print(f"Inner join:\n{concat_inner}")
```

## Horizontal Concatenation (Side by Side)

```python
print("\n=== HORIZONTAL CONCATENATION ===")

# Customer info
customer_info = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
}, index=[1, 2, 3])

# Customer scores
customer_scores = pd.DataFrame({
    'credit_score': [750, 680, 720],
    'loyalty_score': [85, 70, 90]
}, index=[1, 2, 3])

print(f"Customer info:\n{customer_info}\n")
print(f"Customer scores:\n{customer_scores}\n")

# Concatenate horizontally (axis=1)
combined = pd.concat([customer_info, customer_scores], axis=1)
print(f"Combined (horizontal):\n{combined}")
```

## Filtering Joins: Semi-Join

```python
print("\n=== SEMI-JOIN (Filtering Join) ===")
print("Keep rows from left table WHERE key exists in right table")

# All customers
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'tier': ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver']
})

# Customers who made purchases
purchases = pd.DataFrame({
    'customer_id': [1, 2, 2, 3],
    'amount': [100, 50, 75, 200]
})

print(f"All customers:\n{customers}\n")
print(f"Purchases:\n{purchases}\n")

# Semi-join: customers who made at least one purchase
# Method 1: Using isin()
active_customers = customers[customers['customer_id'].isin(purchases['customer_id'])]
print(f"Semi-join (customers with purchases):\n{active_customers}\n")

# Method 2: Using merge with indicator
merged = pd.merge(customers, purchases[['customer_id']].drop_duplicates(), 
                  on='customer_id', how='left', indicator=True)
active_customers_v2 = merged[merged['_merge'] == 'both'].drop(columns='_merge')
print(f"Semi-join (method 2):\n{active_customers_v2}")
```

## Filtering Joins: Anti-Join

```python
print("\n=== ANTI-JOIN ===")
print("Keep rows from left table WHERE key does NOT exist in right table")

# Anti-join: customers who have NOT made any purchases
# Method 1: Using ~isin()
inactive_customers = customers[~customers['customer_id'].isin(purchases['customer_id'])]
print(f"Anti-join (customers without purchases):\n{inactive_customers}\n")

# Method 2: Using merge with indicator
merged = pd.merge(customers, purchases[['customer_id']].drop_duplicates(), 
                  on='customer_id', how='left', indicator=True)
inactive_customers_v2 = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
print(f"Anti-join (method 2):\n{inactive_customers_v2}")
```

## Practical Semi-Join and Anti-Join Functions

```python
print("\n=== REUSABLE FUNCTIONS ===")

def semi_join(left_df, right_df, on):
    """Keep rows from left where key exists in right."""
    return left_df[left_df[on].isin(right_df[on])]

def anti_join(left_df, right_df, on):
    """Keep rows from left where key does NOT exist in right."""
    return left_df[~left_df[on].isin(right_df[on])]

# Example usage
products = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
    'product_name': ['Widget', 'Gadget', 'Gizmo', 'Thing', 'Stuff'],
    'price': [10, 20, 15, 25, 30]
})

sold_products = pd.DataFrame({
    'product_id': ['P1', 'P3', 'P3', 'P5'],
    'quantity': [5, 3, 2, 8]
})

print(f"Products:\n{products}\n")
print(f"Sales:\n{sold_products}\n")

# Products that have been sold
print(f"Sold products (semi-join):\n{semi_join(products, sold_products, 'product_id')}\n")

# Products that have NOT been sold
print(f"Unsold products (anti-join):\n{anti_join(products, sold_products, 'product_id')}")
```

## Appending DataFrames

```python
print("\n=== APPENDING DATA ===")

# Base data
base_df = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [100, 200, 300]
})

# New data to append
new_data = pd.DataFrame({
    'id': [4, 5],
    'value': [400, 500]
})

print(f"Base:\n{base_df}\n")
print(f"New data:\n{new_data}\n")

# Append using concat
updated_df = pd.concat([base_df, new_data], ignore_index=True)
print(f"After appending:\n{updated_df}")
```

## Combining Multiple Operations

```python
print("\n=== COMBINING OPERATIONS ===")

# Multiple data sources
online_sales = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P3'],
    'channel': 'online',
    'sales': [1000, 1500, 800]
})

store_sales = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P4'],
    'channel': 'store',
    'sales': [500, 700, 600]
})

# Concatenate all sales
all_sales = pd.concat([online_sales, store_sales], ignore_index=True)
print(f"All sales:\n{all_sales}\n")

# Products sold online only (anti-join)
online_only = anti_join(
    online_sales[['product_id']].drop_duplicates(),
    store_sales[['product_id']].drop_duplicates(),
    'product_id'
)
print(f"Products sold online only:\n{online_only}\n")

# Products sold in both channels (semi-join both ways)
both_channels = semi_join(
    online_sales[['product_id']].drop_duplicates(),
    store_sales[['product_id']].drop_duplicates(),
    'product_id'
)
print(f"Products sold in both channels:\n{both_channels}")
```

## Key Points

- **concat()**: Stack DataFrames vertically (axis=0) or horizontally (axis=1)
- **ignore_index**: Reset index after concatenation
- **keys**: Add hierarchical index to identify source DataFrames
- **Semi-join**: Filter left table to rows with matches in right (`isin()`)
- **Anti-join**: Filter left table to rows WITHOUT matches in right (`~isin()`)
- **join parameter**: 'outer' keeps all columns, 'inner' keeps common only

## Reflection Questions

1. When would you use concatenation vs merge/join?
2. How do semi-joins and anti-joins differ from regular joins?
3. What's the advantage of using keys parameter in concat?
