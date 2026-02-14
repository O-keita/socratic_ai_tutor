# Introduction to Time Series

## Introduction

Time series data is sequences of observations collected over time. Understanding time series is essential for forecasting, anomaly detection, and analyzing temporal patterns in domains like finance, healthcare, and IoT.

## What is Time Series Data?

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

print("=== TIME SERIES DATA ===")
print("""
A TIME SERIES is a sequence of data points indexed by time:
  x = [x₁, x₂, x₃, ..., xₜ]

Key characteristics:
  1. ORDERED: Sequence matters (unlike i.i.d. data)
  2. TEMPORAL: Time intervals between observations
  3. DEPENDENT: Current values depend on past values

Examples:
  - Stock prices (daily, hourly, minute-by-minute)
  - Temperature readings
  - Website traffic
  - Heart rate monitoring
  - Sales data
""")
```

## Creating Time Series in Pandas

```python
print("\n=== TIME SERIES IN PANDAS ===")

# Create date range
dates = pd.date_range(start='2024-01-01', periods=365, freq='D')

# Generate sample time series (with trend, seasonality, noise)
np.random.seed(42)
t = np.arange(365)
trend = 0.1 * t  # Upward trend
seasonality = 10 * np.sin(2 * np.pi * t / 365)  # Annual cycle
weekly = 5 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
noise = np.random.randn(365) * 3  # Random noise

values = 100 + trend + seasonality + weekly + noise

# Create time series DataFrame
ts = pd.Series(values, index=dates, name='value')

print("Sample time series (first 10 days):")
print(ts.head(10))

print(f"\nTime series info:")
print(f"  Start: {ts.index[0]}")
print(f"  End: {ts.index[-1]}")
print(f"  Frequency: {ts.index.freq}")
print(f"  Length: {len(ts)}")
```

## Time Series Components

```python
print("\n=== TIME SERIES COMPONENTS ===")
print("""
A time series can be decomposed into:

1. TREND (T):
   - Long-term increase or decrease
   - General direction over time

2. SEASONALITY (S):
   - Regular, periodic patterns
   - Daily, weekly, monthly, yearly cycles

3. CYCLICAL (C):
   - Long-term fluctuations
   - Not fixed period (unlike seasonality)
   - Economic cycles

4. RESIDUAL/NOISE (R):
   - Random variation
   - What's left after removing other components

Decomposition models:
  ADDITIVE: Y = T + S + C + R
    (Components add up)
    
  MULTIPLICATIVE: Y = T × S × C × R
    (Seasonal effect proportional to trend)
""")

# Simple decomposition demonstration
df = pd.DataFrame({'value': values}, index=dates)

# Calculate rolling mean (trend proxy)
df['trend'] = df['value'].rolling(window=30, center=True).mean()

# Detrended (for seasonal)
df['detrended'] = df['value'] - df['trend']

print("Component estimates (sample):")
print(df[['value', 'trend', 'detrended']].dropna().head(10).round(2))
```

## Stationarity

```python
print("\n=== STATIONARITY ===")
print("""
A time series is STATIONARY if its statistical properties don't change over time:
  - Constant mean
  - Constant variance
  - Covariance depends only on lag (not time)

WHY IT MATTERS:
  - Many models assume stationarity
  - Easier to model and forecast
  - Properties learned from past apply to future

TYPES:
  - Strict stationarity: All moments unchanged
  - Weak stationarity: Mean and variance unchanged

HOW TO ACHIEVE:
  1. Differencing: Xₜ' = Xₜ - Xₜ₋₁
  2. Log transform: For multiplicative seasonality
  3. Detrending: Remove trend component
""")

# Example: Making series stationary
print("Making series stationary with differencing:")

# First difference
ts_diff = ts.diff().dropna()

print(f"\nOriginal series:")
print(f"  Mean: {ts.mean():.2f}")
print(f"  Std: {ts.std():.2f}")

print(f"\nDifferenced series:")
print(f"  Mean: {ts_diff.mean():.2f}")
print(f"  Std: {ts_diff.std():.2f}")

print("\nDifferencing removed the trend (mean closer to 0)")
```

## Autocorrelation

```python
print("\n=== AUTOCORRELATION ===")
print("""
AUTOCORRELATION: Correlation of series with its lagged version

ACF (Autocorrelation Function):
  - Correlation at different lags
  - Includes indirect correlations

PACF (Partial Autocorrelation Function):
  - Direct correlation at each lag
  - Removes influence of intermediate lags

Uses:
  - Identify patterns (seasonality)
  - Model selection (ARIMA parameters)
  - Check residuals (should be uncorrelated)
""")

# Calculate autocorrelation
def autocorrelation(series, lag):
    """Calculate autocorrelation at given lag"""
    n = len(series)
    mean = series.mean()
    var = series.var()
    
    cov = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
    return cov / var if var > 0 else 0

print("Autocorrelation at different lags:")
for lag in [1, 7, 14, 30, 365]:
    if lag < len(ts):
        acf = autocorrelation(ts.values, lag)
        print(f"  Lag {lag:3d}: {acf:.3f}")

print("""
Interpretation:
  - High ACF at lag 7: Weekly pattern
  - High ACF at lag 365: Annual pattern
  - Decaying ACF: Trend present
""")
```

## Resampling and Aggregation

```python
print("\n=== RESAMPLING AND AGGREGATION ===")
print("""
Common operations:
  - Downsampling: Higher to lower frequency (daily → monthly)
  - Upsampling: Lower to higher frequency (monthly → daily)
  - Aggregation: Sum, mean, min, max over periods
""")

# Different resamplings
print("Weekly mean:")
ts_weekly = ts.resample('W').mean()
print(ts_weekly.head())

print("\nMonthly aggregations:")
ts_monthly = ts.resample('ME').agg(['mean', 'min', 'max', 'std'])
print(ts_monthly.head())

print("\nQuarterly sum:")
ts_quarterly = ts.resample('QE').sum()
print(ts_quarterly)
```

## Rolling Windows

```python
print("\n=== ROLLING WINDOWS ===")
print("""
Rolling (moving) calculations over a sliding window:
  - Moving average: Smooth out noise
  - Moving std: Volatility/variance over time
  - Moving sum: Cumulative metrics

Common uses:
  - Smoothing noisy data
  - Identifying trends
  - Technical analysis (finance)
""")

# Different rolling windows
df_roll = pd.DataFrame({'value': ts})

df_roll['MA7'] = ts.rolling(window=7).mean()   # 7-day moving average
df_roll['MA30'] = ts.rolling(window=30).mean() # 30-day moving average
df_roll['STD7'] = ts.rolling(window=7).std()   # 7-day rolling std

print("Rolling statistics (sample):")
print(df_roll.iloc[30:40].round(2))

print("""
Note: First (window-1) values are NaN (not enough data)
""")
```

## Time Series Train/Test Split

```python
print("\n=== TIME SERIES TRAIN/TEST SPLIT ===")
print("""
IMPORTANT: Cannot randomly split time series!

Why:
  - Would leak future information to training
  - Violates temporal order
  
Correct approach:
  - Use earlier data for training
  - Use later data for testing
  - Never look ahead!
""")

# Correct split
train_size = int(len(ts) * 0.8)
ts_train = ts[:train_size]
ts_test = ts[train_size:]

print(f"Training set: {ts_train.index[0]} to {ts_train.index[-1]} ({len(ts_train)} points)")
print(f"Test set: {ts_test.index[0]} to {ts_test.index[-1]} ({len(ts_test)} points)")

print("""
For cross-validation, use:
  - Expanding window: Train on 1:t, test on t+1
  - Rolling window: Train on t-k:t, test on t+1
  
sklearn: TimeSeriesSplit
""")

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
print("\nTimeSeriesSplit folds:")
for i, (train_idx, test_idx) in enumerate(tscv.split(ts)):
    print(f"  Fold {i+1}: Train {len(train_idx)}, Test {len(test_idx)}")
```

## Key Points

- **Time series**: Ordered sequence of observations over time
- **Components**: Trend, seasonality, cyclical, residual
- **Stationarity**: Constant statistical properties over time
- **Autocorrelation**: Correlation with lagged versions
- **Differencing**: Common technique to achieve stationarity
- **Rolling windows**: Moving statistics for smoothing/analysis
- **Train/test split**: Must respect temporal order (no shuffling!)

## Reflection Questions

1. Why can't we randomly shuffle time series data for cross-validation?
2. What would happen if you apply a model trained on stationary data to non-stationary test data?
3. How would you detect if a time series has weekly seasonality?
