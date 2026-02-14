# Left and Right Joins in Pandas

## Introduction

Left and right joins are asymmetric join operations that preserve all rows from one table while matching rows from another. These are commonly used when you have a primary table and want to enrich it with optional data from a secondary table.

## Left Join

```python
import pandas as pd
import numpy as np

# Sample data
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'department': ['Sales', 'Engineering', 'Sales', 'HR', 'Engineering']
})

performance = pd.DataFrame({
    'emp_id': [1, 2, 3, 6],  # Note: emp 4, 5 missing; emp 6 extra
    'rating': [4.5, 4.0, 3.8, 4.2],
    'bonus': [1000, 800, 600, 900]
})

print("=== LEFT JOIN ===")
print(f"Employees (left table):\n{employees}\n")
print(f"Performance (right table):\n{performance}\n")

# Left join: keep ALL employees, add performance where available
left_result = pd.merge(employees, performance, on='emp_id', how='left')
print(f"Left join result:\n{left_result}\n")

print("""
Observations:
  - ALL 5 employees are kept
  - Diana (4) and Eve (5) have NaN for rating/bonus (no performance data)
  - Employee 6 from performance table is NOT included
  - Left table is the "primary" table
""")
```

## Right Join

```python
print("\n=== RIGHT JOIN ===")

# Right join: keep ALL performance records, add employee info where available
right_result = pd.merge(employees, performance, on='emp_id', how='right')
print(f"Right join result:\n{right_result}\n")

print("""
Observations:
  - ALL 4 performance records are kept
  - Employee 6 has NaN for name/department (not in employees table)
  - Employees 4, 5 are NOT included (no performance records)
  - Right table is the "primary" table
""")

# Note: left join with swapped tables = right join
# These are equivalent:
# pd.merge(A, B, how='left') == pd.merge(B, A, how='right')
```

## Left Join: Preserving Primary Data

```python
print("\n=== PRESERVING PRIMARY DATA ===")

# Products (primary table - all products must be in result)
products = pd.DataFrame({
    'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'product_name': ['Widget', 'Gadget', 'Gizmo', 'Thingamajig', 'Doohickey'],
    'category': ['Electronics', 'Electronics', 'Home', 'Home', 'Office']
})

# Sales this month (optional data - not all products sold)
monthly_sales = pd.DataFrame({
    'product_id': ['P001', 'P003', 'P003', 'P005'],
    'quantity': [10, 5, 3, 15],
    'revenue': [100, 75, 45, 150]
})

print(f"All products:\n{products}\n")
print(f"Monthly sales:\n{monthly_sales}\n")

# Left join to see all products with their sales (if any)
product_sales = pd.merge(products, monthly_sales, on='product_id', how='left')
print(f"Products with sales:\n{product_sales}\n")

# Fill NaN with 0 for products with no sales
product_sales_filled = product_sales.fillna({'quantity': 0, 'revenue': 0})
print(f"After filling NaN with 0:\n{product_sales_filled}")
```

## Aggregating Before Join

```python
print("\n=== AGGREGATE THEN JOIN ===")

# First, aggregate sales by product
sales_summary = monthly_sales.groupby('product_id').agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index()
print(f"Sales summary:\n{sales_summary}\n")

# Then left join to products
product_totals = pd.merge(products, sales_summary, on='product_id', how='left')
product_totals = product_totals.fillna({'quantity': 0, 'revenue': 0})
print(f"Product totals:\n{product_totals}")
```

## Multiple Join Keys

```python
print("\n=== MULTIPLE JOIN KEYS ===")

# Store locations
stores = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South'],
    'store_id': [1, 2, 1, 2],
    'store_name': ['NYC Main', 'Boston', 'Miami', 'Atlanta']
})

# Regional sales
regional_sales = pd.DataFrame({
    'region': ['North', 'North', 'South', 'West'],
    'store_id': [1, 2, 1, 1],
    'monthly_sales': [50000, 35000, 42000, 28000]
})

print(f"Stores:\n{stores}\n")
print(f"Regional sales:\n{regional_sales}\n")

# Left join on multiple keys
store_performance = pd.merge(
    stores, regional_sales, 
    on=['region', 'store_id'], 
    how='left'
)
print(f"Store performance:\n{store_performance}")
```

## Checking Join Quality

```python
print("\n=== CHECKING JOIN QUALITY ===")

# Use indicator to understand join results
result = pd.merge(employees, performance, on='emp_id', how='left', indicator=True)
print(f"Left join with indicator:\n{result}\n")

# Count matches
match_counts = result['_merge'].value_counts()
print(f"Match summary:\n{match_counts}\n")

# Percentage matched
pct_matched = (result['_merge'] == 'both').sum() / len(result) * 100
print(f"Percentage of employees with performance data: {pct_matched:.1f}%")

# Find unmatched
unmatched = result[result['_merge'] == 'left_only']
print(f"\nEmployees without performance data:\n{unmatched[['emp_id', 'name']]}")
```

## One-to-Many Joins

```python
print("\n=== ONE-TO-MANY JOINS ===")

# Departments (one)
departments = pd.DataFrame({
    'dept_id': [1, 2, 3],
    'dept_name': ['Sales', 'Engineering', 'HR']
})

# Employees (many)
employees_dept = pd.DataFrame({
    'emp_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'dept_id': [1, 2, 1, 3, 2]
})

print(f"Departments:\n{departments}\n")
print(f"Employees:\n{employees_dept}\n")

# Left join: all employees with their department names
emp_with_dept = pd.merge(employees_dept, departments, on='dept_id', how='left')
print(f"Employees with department names:\n{emp_with_dept}\n")

# Right join: all departments with their employees
dept_with_emp = pd.merge(employees_dept, departments, on='dept_id', how='right')
print(f"Departments with employees:\n{dept_with_emp}")
```

## Practical Example: Customer Analysis

```python
print("\n=== PRACTICAL EXAMPLE ===")

# All customers
customers = pd.DataFrame({
    'customer_id': [101, 102, 103, 104, 105],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'signup_date': pd.to_datetime(['2023-01', '2023-03', '2023-06', '2023-09', '2024-01'])
})

# Recent orders (only some customers)
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4, 5],
    'customer_id': [101, 101, 102, 103, 103],
    'order_date': pd.to_datetime(['2024-01-15', '2024-01-20', '2024-01-18', '2024-01-22', '2024-01-25']),
    'amount': [150, 200, 75, 300, 125]
})

print(f"Customers:\n{customers}\n")
print(f"Orders:\n{orders}\n")

# Step 1: Aggregate orders per customer
customer_orders = orders.groupby('customer_id').agg({
    'order_id': 'count',
    'amount': 'sum',
    'order_date': 'max'
}).rename(columns={
    'order_id': 'num_orders',
    'amount': 'total_spent',
    'order_date': 'last_order'
}).reset_index()

# Step 2: Left join to get all customers
customer_analysis = pd.merge(customers, customer_orders, on='customer_id', how='left')

# Step 3: Fill NaN and add derived columns
customer_analysis['num_orders'] = customer_analysis['num_orders'].fillna(0).astype(int)
customer_analysis['total_spent'] = customer_analysis['total_spent'].fillna(0)
customer_analysis['is_active'] = customer_analysis['num_orders'] > 0

print(f"Customer analysis:\n{customer_analysis}")
```

## Key Points

- **Left join**: All rows from left table + matching from right
- **Right join**: All rows from right table + matching from left
- **Left join preferred**: Most common, keeps primary table intact
- **NaN handling**: Use `fillna()` for missing values after join
- **Indicator**: Shows match status for each row
- **Aggregate first**: Often useful to aggregate before joining

## Reflection Questions

1. Why is left join more commonly used than right join?
2. How would you identify data quality issues using join indicators?
3. When should you aggregate data before performing a join?
