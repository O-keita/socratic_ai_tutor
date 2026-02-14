# Time Series Decomposition

## Introduction

Time series decomposition separates a time series into its constituent components: trend, seasonality, and residuals. This helps understand the underlying patterns and improve forecasting.

## Decomposition Models

```python
import numpy as np
import pandas as pd

np.random.seed(42)

print("=== TIME SERIES DECOMPOSITION ===")
print("""
DECOMPOSITION separates time series into components:
  Yₜ = f(Trendₜ, Seasonalₜ, Residualₜ)

Two main models:

ADDITIVE MODEL:
  Yₜ = Tₜ + Sₜ + Rₜ
  
  Use when:
  - Seasonal fluctuations are constant over time
  - Variations don't grow with level

MULTIPLICATIVE MODEL:
  Yₜ = Tₜ × Sₜ × Rₜ
  
  Use when:
  - Seasonal fluctuations proportional to trend
  - Larger values have larger variations
  
Tip: Log transform converts multiplicative to additive:
  log(Y) = log(T) + log(S) + log(R)
""")

# Create sample data
dates = pd.date_range('2021-01-01', periods=730, freq='D')
t = np.arange(730)

# Components
trend = 100 + 0.15 * t
seasonal = 20 * np.sin(2 * np.pi * t / 365.25)
residual = np.random.randn(730) * 5

# Additive model
y_additive = trend + seasonal + residual

# Multiplicative model (seasonal effect proportional to trend)
y_multiplicative = trend * (1 + 0.2 * np.sin(2 * np.pi * t / 365.25)) * (1 + np.random.randn(730) * 0.03)

ts_add = pd.Series(y_additive, index=dates)
ts_mult = pd.Series(y_multiplicative, index=dates)

print("Created additive and multiplicative time series")
print(f"  Additive range: [{y_additive.min():.1f}, {y_additive.max():.1f}]")
print(f"  Multiplicative range: [{y_multiplicative.min():.1f}, {y_multiplicative.max():.1f}]")
```

## Classical Decomposition

```python
print("\n=== CLASSICAL DECOMPOSITION ===")
print("""
Steps for ADDITIVE decomposition:

1. TREND: Moving average to smooth out seasonal + noise
   - For monthly data with yearly cycle: 12-MA
   - For daily data with weekly cycle: 7-MA

2. DETREND: Subtract trend from original
   detrended = Y - Trend

3. SEASONAL: Average detrended values for each season
   - Average all Januaries, all Februaries, etc.
   - Or average all Mondays, all Tuesdays, etc.

4. RESIDUAL: What remains
   Residual = Y - Trend - Seasonal
""")

def classical_decomposition(series, period):
    """Simple classical decomposition"""
    # 1. Trend (centered moving average)
    trend = series.rolling(window=period, center=True).mean()
    
    # 2. Detrended
    detrended = series - trend
    
    # 3. Seasonal (average for each period position)
    seasonal_values = []
    for i in range(len(series)):
        position = i % period
        same_position = [detrended.iloc[j] for j in range(len(detrended)) 
                        if j % period == position and not np.isnan(detrended.iloc[j])]
        seasonal_values.append(np.mean(same_position))
    seasonal = pd.Series(seasonal_values, index=series.index)
    
    # 4. Residual
    residual = series - trend - seasonal
    
    return trend, seasonal, residual

# Apply to additive series
trend, seasonal, residual = classical_decomposition(ts_add, period=365)

print("Classical decomposition results (sample days):")
result_df = pd.DataFrame({
    'Original': ts_add,
    'Trend': trend,
    'Seasonal': seasonal,
    'Residual': residual
})
print(result_df.iloc[200:210].round(2))
```

## STL Decomposition

```python
print("\n=== STL DECOMPOSITION ===")
print("""
STL = Seasonal and Trend decomposition using Loess

Advantages over classical:
  - Handles any type of seasonality
  - Robust to outliers
  - Seasonal component can change over time
  - More flexible

Components:
  - trend: Loess-smoothed trend
  - seasonal: Seasonal component
  - resid: Residual
""")

from statsmodels.tsa.seasonal import STL

# STL decomposition
stl = STL(ts_add, period=365, robust=True)
result = stl.fit()

print("STL decomposition components:")
print(f"  Trend range: [{result.trend.min():.1f}, {result.trend.max():.1f}]")
print(f"  Seasonal range: [{result.seasonal.min():.1f}, {result.seasonal.max():.1f}]")
print(f"  Residual range: [{result.resid.min():.1f}, {result.resid.max():.1f}]")

# Verify: original ≈ trend + seasonal + residual
reconstructed = result.trend + result.seasonal + result.resid
error = (ts_add - reconstructed).abs().max()
print(f"\nReconstruction error: {error:.10f} (should be ~0)")
```

## Seasonal Decompose in Statsmodels

```python
print("\n=== SEASONAL_DECOMPOSE ===")

from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
decomp_add = seasonal_decompose(ts_add, model='additive', period=365)

print("Additive decomposition:")
print(f"  Observed: {len(decomp_add.observed)} values")
print(f"  Trend: {decomp_add.trend.dropna().mean():.2f} mean")
print(f"  Seasonal: {decomp_add.seasonal.std():.2f} std")
print(f"  Residual: {decomp_add.resid.dropna().std():.2f} std")

# Multiplicative decomposition
decomp_mult = seasonal_decompose(ts_mult, model='multiplicative', period=365)

print("\nMultiplicative decomposition:")
print(f"  Trend: {decomp_mult.trend.dropna().mean():.2f} mean")
print(f"  Seasonal: {decomp_mult.seasonal.mean():.3f} mean (should be ~1)")
print(f"  Residual: {decomp_mult.resid.dropna().std():.4f} std")
```

## Choosing Additive vs Multiplicative

```python
print("\n=== CHOOSING THE MODEL ===")
print("""
How to decide between additive and multiplicative:

1. VISUAL INSPECTION:
   - Plot the series
   - If seasonal amplitude grows with level → multiplicative
   - If seasonal amplitude constant → additive

2. COEFFICIENT OF VARIATION:
   - Calculate std/mean for different periods
   - If relatively constant → additive
   - If increases with mean → multiplicative

3. LOG TRANSFORM TEST:
   - Take log of data
   - If logged series has constant seasonal amplitude → multiplicative

4. RULE OF THUMB:
   - Count data (visitors, sales) → often multiplicative
   - Temperature, measurement → often additive
""")

# Check seasonal amplitude over time
def check_amplitude(series, period):
    """Calculate seasonal amplitude in different periods"""
    n_periods = len(series) // period
    amplitudes = []
    means = []
    
    for i in range(n_periods):
        segment = series.iloc[i*period:(i+1)*period]
        amplitudes.append(segment.max() - segment.min())
        means.append(segment.mean())
    
    return means, amplitudes

# Test on additive
means_add, amps_add = check_amplitude(ts_add, 365)
print("Additive series amplitude vs mean:")
for m, a in zip(means_add[:3], amps_add[:3]):
    print(f"  Mean={m:.1f}, Amplitude={a:.1f}")

# Test on multiplicative
means_mult, amps_mult = check_amplitude(ts_mult, 365)
print("\nMultiplicative series amplitude vs mean:")
for m, a in zip(means_mult[:3], amps_mult[:3]):
    print(f"  Mean={m:.1f}, Amplitude={a:.1f}")

print("\nNote: Multiplicative amplitude grows with mean!")
```

## Using Decomposition for Forecasting

```python
print("\n=== DECOMPOSITION FOR FORECASTING ===")
print("""
Decomposition helps forecasting:

1. TREND EXTRAPOLATION:
   - Fit line/polynomial to trend
   - Extrapolate to future

2. SEASONAL PATTERN:
   - Use average seasonal factors
   - Apply to future periods

3. RECOMBINE:
   - Additive: Forecast = Trend + Seasonal
   - Multiplicative: Forecast = Trend × Seasonal

Residual analysis:
   - Check for remaining patterns
   - Should be approximately white noise
""")

# Simple forecast example
def simple_forecast(series, period, horizon):
    """Naive decomposition-based forecast"""
    # Decompose
    decomp = seasonal_decompose(series, model='additive', period=period)
    
    # Trend: linear extrapolation
    trend_valid = decomp.trend.dropna()
    x = np.arange(len(trend_valid))
    slope, intercept = np.polyfit(x, trend_valid.values, 1)
    
    # Forecast trend
    x_future = np.arange(len(trend_valid), len(trend_valid) + horizon)
    trend_forecast = slope * x_future + intercept
    
    # Seasonal factors (average pattern)
    seasonal_pattern = decomp.seasonal.iloc[:period].values
    seasonal_forecast = np.tile(seasonal_pattern, (horizon // period) + 1)[:horizon]
    
    # Combine
    forecast = trend_forecast + seasonal_forecast
    return forecast

# Make forecast
horizon = 30
forecast = simple_forecast(ts_add, period=365, horizon=horizon)

print(f"30-day forecast (first 10 days):")
for i in range(10):
    print(f"  Day {i+1}: {forecast[i]:.2f}")
```

## Key Points

- **Decomposition**: Separates time series into trend, seasonal, residual
- **Additive**: Y = T + S + R (constant seasonal amplitude)
- **Multiplicative**: Y = T × S × R (proportional seasonal effect)
- **Classical**: Simple moving average approach
- **STL**: More robust, handles changing seasonality
- **Model selection**: Check if amplitude grows with level
- **Forecasting**: Extrapolate components and recombine

## Reflection Questions

1. How would you identify whether a time series is additive or multiplicative?
2. What are the limitations of using decomposition for long-term forecasting?
3. How might you handle multiple seasonal patterns (e.g., daily AND yearly)?
