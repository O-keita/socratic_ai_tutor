# Inner and Outer Joins in Pandas

## Introduction

Joining tables is a fundamental operation in data analysis. Inner and outer joins allow you to combine data from multiple DataFrames based on common columns (keys), similar to SQL joins.

## Understanding Joins

```python
import pandas as pd
import numpy as np

print("=== UNDERSTANDING JOINS ===")
print("""
Join Types:
  INNER JOIN: Only matching rows from both tables
  LEFT JOIN:  All rows from left + matches from right
  RIGHT JOIN: All rows from right + matches from left
  OUTER JOIN: All rows from both tables

Visual Representation:
  
  Left Table    Right Table
  ┌─────┐       ┌─────┐
  │  A  │       │  B  │
  │ ┌───┼───┐   │     │
  │ │ A │ B │   │     │
  │ │ ∩ │   │   │     │
  │ └───┼───┘   │     │
  └─────┘       └─────┘
  
  INNER: A ∩ B (intersection only)
  LEFT:  All A + matching B
  RIGHT: All B + matching A
  OUTER: A ∪ B (everything)
""")
```

## Sample Data Setup

```python
# Create sample DataFrames
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 2, 3, 6],  # Note: customer 6 doesn't exist
    'amount': [150, 200, 75, 300, 125]
})

print("=== SAMPLE DATA ===")
print(f"Customers:\n{customers}\n")
print(f"Orders:\n{orders}")
print("\nNote: Customer IDs 4, 5 have no orders; Order 105 has customer_id 6 (not in customers)")
```

## Inner Join

```python
print("\n=== INNER JOIN ===")
print("Returns only rows where key exists in BOTH tables")

inner_result = pd.merge(customers, orders, on='customer_id', how='inner')
print(f"Inner join result:\n{inner_result}\n")

print("""
Observations:
  - Customers 4, 5 excluded (no matching orders)
  - Order 105 excluded (customer_id 6 not in customers)
  - Bob appears twice (has 2 orders)
  - Only matching rows kept
""")
```

## Outer Join (Full Outer Join)

```python
print("\n=== OUTER JOIN (FULL) ===")
print("Returns ALL rows from BOTH tables, with NaN where no match")

outer_result = pd.merge(customers, orders, on='customer_id', how='outer')
print(f"Outer join result:\n{outer_result}\n")

print("""
Observations:
  - All customers included (4, 5 have NaN for order columns)
  - All orders included (customer_id 6 has NaN for customer columns)
  - NaN indicates missing match
  - Most inclusive join type
""")
```

## Using merge() Parameters

```python
print("\n=== MERGE PARAMETERS ===")

# Different column names
products = pd.DataFrame({
    'prod_id': [1, 2, 3],
    'product_name': ['Widget', 'Gadget', 'Gizmo']
})

sales = pd.DataFrame({
    'sale_id': [1, 2, 3],
    'product_code': [1, 2, 4],  # Note: product 4 doesn't exist
    'quantity': [10, 5, 8]
})

print(f"Products (key: prod_id):\n{products}\n")
print(f"Sales (key: product_code):\n{sales}\n")

# Join on different column names
merged = pd.merge(products, sales, left_on='prod_id', right_on='product_code', how='inner')
print(f"Inner join with different key names:\n{merged}\n")

# Outer join
merged_outer = pd.merge(products, sales, left_on='prod_id', right_on='product_code', how='outer')
print(f"Outer join:\n{merged_outer}")
```

## Handling Duplicate Column Names

```python
print("\n=== HANDLING DUPLICATES ===")

df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [100, 200, 300],
    'date': ['2024-01', '2024-02', '2024-03']
})

df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'value': [150, 250, 350],
    'category': ['A', 'B', 'C']
})

print(f"df1:\n{df1}\n")
print(f"df2:\n{df2}\n")

# Default suffixes
merged = pd.merge(df1, df2, on='id', how='outer')
print(f"Merged (default suffixes _x, _y):\n{merged}\n")

# Custom suffixes
merged_custom = pd.merge(df1, df2, on='id', how='outer', suffixes=('_left', '_right'))
print(f"Merged (custom suffixes):\n{merged_custom}")
```

## Join on Index

```python
print("\n=== JOIN ON INDEX ===")

# Set index
customers_indexed = customers.set_index('customer_id')
orders_indexed = orders.set_index('customer_id')

print(f"Customers (indexed):\n{customers_indexed}\n")
print(f"Orders (indexed):\n{orders_indexed}\n")

# Join on index
joined = customers_indexed.join(orders_indexed, how='inner')
print(f"Inner join on index:\n{joined}\n")

# Using merge with index
merged = pd.merge(customers_indexed, orders_indexed, left_index=True, right_index=True, how='outer')
print(f"Outer merge on index:\n{merged}")
```

## Indicator Column

```python
print("\n=== INDICATOR COLUMN ===")
print("Shows which table each row came from")

merged_indicator = pd.merge(customers, orders, on='customer_id', how='outer', indicator=True)
print(f"With indicator:\n{merged_indicator}\n")

# Filter based on indicator
left_only = merged_indicator[merged_indicator['_merge'] == 'left_only']
print(f"Customers with no orders:\n{left_only[['customer_id', 'name']]}\n")

right_only = merged_indicator[merged_indicator['_merge'] == 'right_only']
print(f"Orders with no matching customer:\n{right_only[['order_id', 'customer_id']]}")
```

## Practical Example: E-commerce Analysis

```python
print("\n=== PRACTICAL EXAMPLE ===")

# Customer data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'segment': ['Premium', 'Standard', 'Premium', 'Standard', 'Premium']
})

# Order data
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 1, 2, 3, 3, 3],
    'amount': [150, 200, 75, 300, 125, 50]
})

# Inner join to analyze ordering customers
active_customers = pd.merge(customers, orders, on='customer_id', how='inner')
print(f"Active customers with orders:\n{active_customers}\n")

# Aggregate by customer
customer_totals = active_customers.groupby(['customer_id', 'name', 'segment']).agg({
    'order_id': 'count',
    'amount': 'sum'
}).rename(columns={'order_id': 'num_orders', 'amount': 'total_spent'})
print(f"Customer summary:\n{customer_totals}\n")

# Outer join to find inactive customers
all_customers = pd.merge(customers, orders, on='customer_id', how='outer', indicator=True)
inactive = all_customers[all_customers['_merge'] == 'left_only'][['customer_id', 'name', 'segment']]
print(f"Inactive customers (no orders):\n{inactive}")
```

## Key Points

- **Inner join**: Only matching rows from both tables
- **Outer join**: All rows from both, NaN for no match
- **merge()**: Main function with `how='inner'` or `how='outer'`
- **on**: Column(s) to join on
- **left_on/right_on**: For different column names
- **suffixes**: Handle duplicate column names
- **indicator**: Shows source table for each row

## Reflection Questions

1. When would you use an inner join vs an outer join?
2. How does the indicator parameter help with data quality checks?
3. What happens when joining tables with duplicate keys in both?
