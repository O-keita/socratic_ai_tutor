# Exponential Smoothing

## Introduction

Exponential smoothing methods are widely used for time series forecasting. They give more weight to recent observations and can handle trends and seasonality through different model variations.

## Simple Exponential Smoothing

```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=== EXPONENTIAL SMOOTHING ===")
print("""
SIMPLE EXPONENTIAL SMOOTHING (SES):
  Forecast is weighted average of all past observations
  More recent observations get more weight
  
  Formula:
    ŷₜ₊₁ = α * yₜ + (1-α) * ŷₜ
    
  Where:
    α = smoothing parameter (0 < α < 1)
    yₜ = actual value at time t
    ŷₜ = smoothed value (forecast) at time t
    
  Equivalent to:
    ŷₜ₊₁ = α * yₜ + α(1-α) * yₜ₋₁ + α(1-α)² * yₜ₋₂ + ...
    
  Weights decay exponentially!

USE WHEN:
  - No trend
  - No seasonality
  - Level (mean) may change slowly
""")

# Create sample data (no trend)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
np.random.seed(42)
y = 50 + np.random.randn(100) * 5

ts = pd.Series(y, index=dates)
print(f"Created series: mean={ts.mean():.2f}, std={ts.std():.2f}")
```

## Implementing SES

```python
print("\n=== SIMPLE EXPONENTIAL SMOOTHING ===")

def simple_exp_smoothing(series, alpha):
    """Simple exponential smoothing"""
    n = len(series)
    smoothed = np.zeros(n)
    smoothed[0] = series.iloc[0]  # Initialize with first value
    
    for t in range(1, n):
        smoothed[t] = alpha * series.iloc[t-1] + (1 - alpha) * smoothed[t-1]
    
    return pd.Series(smoothed, index=series.index)

# Compare different alpha values
print("Effect of alpha parameter:")
print(f"{'Alpha':>6} | {'MAE':>8} | Interpretation")
print("-" * 50)

for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    smoothed = simple_exp_smoothing(ts, alpha)
    mae = (ts - smoothed).abs().mean()
    if alpha < 0.3:
        interp = "Smooth (slow adaptation)"
    elif alpha < 0.7:
        interp = "Balanced"
    else:
        interp = "Responsive (fast adaptation)"
    print(f"{alpha:>6} | {mae:>8.3f} | {interp}")

print("""
Low alpha: Smooth forecast, slow to adapt
High alpha: Noisy forecast, quick to adapt
""")
```

## Using Statsmodels

```python
print("\n=== USING STATSMODELS ===")

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Fit SES model
model = SimpleExpSmoothing(ts)
fitted = model.fit(smoothing_level=0.3, optimized=False)

print(f"SES with alpha=0.3:")
print(f"  Smoothing level: {fitted.params['smoothing_level']:.3f}")
print(f"  AIC: {fitted.aic:.1f}")

# Optimal alpha
fitted_opt = model.fit(optimized=True)
print(f"\nSES with optimal alpha:")
print(f"  Optimal alpha: {fitted_opt.params['smoothing_level']:.3f}")
print(f"  AIC: {fitted_opt.aic:.1f}")

# Forecast
forecast = fitted_opt.forecast(10)
print(f"\n10-step forecast (constant level):")
print(forecast.round(2).values)
```

## Holt's Linear Trend Method

```python
print("\n=== HOLT'S LINEAR TREND METHOD ===")
print("""
Extends SES to handle linear trends.

Two equations:
  Level: lₜ = α * yₜ + (1-α) * (lₜ₋₁ + bₜ₋₁)
  Trend: bₜ = β * (lₜ - lₜ₋₁) + (1-β) * bₜ₋₁

Forecast: ŷₜ₊ₕ = lₜ + h * bₜ

Parameters:
  α = level smoothing (0 < α < 1)
  β = trend smoothing (0 < β < 1)
""")

# Create data with trend
np.random.seed(42)
t = np.arange(100)
y_trend = 50 + 0.5 * t + np.random.randn(100) * 5
ts_trend = pd.Series(y_trend, index=dates)

print(f"Created trend series: start={y_trend[0]:.1f}, end={y_trend[-1]:.1f}")

# Fit Holt's method
model_holt = ExponentialSmoothing(ts_trend, trend='add', seasonal=None)
fitted_holt = model_holt.fit()

print(f"\nHolt's Linear Trend:")
print(f"  Alpha (level): {fitted_holt.params['smoothing_level']:.3f}")
print(f"  Beta (trend): {fitted_holt.params['smoothing_trend']:.3f}")

# Forecast
forecast_holt = fitted_holt.forecast(10)
print(f"\n10-step forecast (with trend):")
print(forecast_holt.round(2).values)
```

## Damped Trend

```python
print("\n=== DAMPED TREND ===")
print("""
Long-term forecasts with linear trend can be unrealistic.
Damped trend flattens out over time.

Forecast: ŷₜ₊ₕ = lₜ + (φ + φ² + ... + φʰ) * bₜ

Where φ (phi) is damping parameter (0 < φ < 1)

As h → ∞, forecast approaches lₜ + φ/(1-φ) * bₜ
""")

# Fit damped trend model
model_damped = ExponentialSmoothing(ts_trend, trend='add', damped_trend=True)
fitted_damped = model_damped.fit()

print(f"Damped Trend:")
print(f"  Alpha: {fitted_damped.params['smoothing_level']:.3f}")
print(f"  Beta: {fitted_damped.params['smoothing_trend']:.3f}")
print(f"  Phi (damping): {fitted_damped.params['damping_trend']:.3f}")

# Compare forecasts
fc_linear = fitted_holt.forecast(30)
fc_damped = fitted_damped.forecast(30)

print(f"\nLong-term forecast comparison:")
print(f"{'Horizon':>8} | {'Linear':>10} | {'Damped':>10}")
print("-" * 35)
for h in [1, 5, 10, 20, 30]:
    print(f"{h:>8} | {fc_linear.iloc[h-1]:>10.1f} | {fc_damped.iloc[h-1]:>10.1f}")
```

## Holt-Winters Seasonal Method

```python
print("\n=== HOLT-WINTERS SEASONAL METHOD ===")
print("""
Extends Holt's method to include seasonality.

Three equations:
  Level: lₜ = α(yₜ - sₜ₋ₘ) + (1-α)(lₜ₋₁ + bₜ₋₁)
  Trend: bₜ = β(lₜ - lₜ₋₁) + (1-β)bₜ₋₁
  Seasonal: sₜ = γ(yₜ - lₜ) + (1-γ)sₜ₋ₘ

Parameters:
  α = level smoothing
  β = trend smoothing
  γ = seasonal smoothing
  m = seasonal period

Models:
  - ADDITIVE: yₜ = lₜ + bₜ + sₜ + εₜ
  - MULTIPLICATIVE: yₜ = lₜ * bₜ * sₜ * εₜ
""")

# Create seasonal data
np.random.seed(42)
t = np.arange(200)
trend = 50 + 0.1 * t
seasonal = 10 * np.sin(2 * np.pi * t / 30)  # Period 30
noise = np.random.randn(200) * 3
y_seasonal = trend + seasonal + noise

dates_long = pd.date_range('2024-01-01', periods=200, freq='D')
ts_seasonal = pd.Series(y_seasonal, index=dates_long)

# Fit Holt-Winters
model_hw = ExponentialSmoothing(ts_seasonal, trend='add', seasonal='add', 
                                seasonal_periods=30)
fitted_hw = model_hw.fit()

print(f"Holt-Winters (Additive):")
print(f"  Alpha (level): {fitted_hw.params['smoothing_level']:.3f}")
print(f"  Beta (trend): {fitted_hw.params['smoothing_trend']:.3f}")
print(f"  Gamma (seasonal): {fitted_hw.params['smoothing_seasonal']:.3f}")

# Forecast
forecast_hw = fitted_hw.forecast(60)
print(f"\nForecasted values show seasonal pattern:")
print(f"  Days 1-5: {forecast_hw.iloc[:5].round(1).values}")
print(f"  Days 15-20: {forecast_hw.iloc[14:20].round(1).values}")
```

## Choosing the Right Method

```python
print("\n=== CHOOSING THE METHOD ===")
print("""
METHOD SELECTION GUIDE:

Data Pattern           | Method
-----------------------+----------------------------------
No trend, no seasonal  | Simple Exponential Smoothing
Linear trend           | Holt's Linear
Damped trend           | Holt's with damping
Trend + seasonality    | Holt-Winters

Seasonal Type:
  Additive:       Y = Level + Seasonal
  Multiplicative: Y = Level × Seasonal

Use multiplicative when seasonal variation grows with level.

PARAMETER SELECTION:
  - Let statsmodels optimize (optimized=True)
  - Or use cross-validation
  - AIC/BIC for model comparison
""")

# Compare models on seasonal data
print("\nModel comparison on seasonal data:")
print(f"{'Model':>25} | {'AIC':>10}")
print("-" * 40)

models = [
    ('SES', SimpleExpSmoothing(ts_seasonal).fit()),
    ('Holt Linear', ExponentialSmoothing(ts_seasonal, trend='add').fit()),
    ('Holt Damped', ExponentialSmoothing(ts_seasonal, trend='add', damped_trend=True).fit()),
    ('HW Additive', ExponentialSmoothing(ts_seasonal, trend='add', seasonal='add', seasonal_periods=30).fit()),
]

for name, model in models:
    print(f"{name:>25} | {model.aic:>10.1f}")

print("\nHolt-Winters has best AIC (captures seasonality)")
```

## Forecast Accuracy

```python
print("\n=== FORECAST ACCURACY ===")

# Split data
train = ts_seasonal[:160]
test = ts_seasonal[160:]

# Fit and forecast
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=30)
fitted = model.fit()
forecast = fitted.forecast(len(test))

# Accuracy metrics
errors = test - forecast
mae = errors.abs().mean()
rmse = np.sqrt((errors ** 2).mean())
mape = (errors.abs() / test.abs()).mean() * 100

print(f"Forecast accuracy (40-step ahead):")
print(f"  MAE: {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAPE: {mape:.2f}%")

# First 10 forecasts
print(f"\nFirst 10 forecasts vs actuals:")
print(f"{'Step':>5} | {'Forecast':>10} | {'Actual':>10} | {'Error':>10}")
print("-" * 45)
for i in range(10):
    print(f"{i+1:>5} | {forecast.iloc[i]:>10.2f} | {test.iloc[i]:>10.2f} | {errors.iloc[i]:>10.2f}")
```

## Key Points

- **SES**: Simple smoothing for no-trend, no-seasonal data
- **Holt**: Adds linear trend component
- **Damped trend**: Flattens trend for long horizons
- **Holt-Winters**: Adds seasonal component
- **Alpha, Beta, Gamma**: Smoothing parameters (0-1)
- **Additive vs Multiplicative**: Based on whether seasonal effect is constant or proportional
- **Optimization**: Let statsmodels find optimal parameters

## Reflection Questions

1. When would you choose exponential smoothing over ARIMA?
2. How does the damping parameter prevent unrealistic long-term forecasts?
3. What would happen if all smoothing parameters were set to 1?
