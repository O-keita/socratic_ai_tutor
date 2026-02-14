# ARIMA Models

## Introduction

ARIMA (AutoRegressive Integrated Moving Average) is one of the most popular statistical methods for time series forecasting. It combines autoregression, differencing, and moving average components.

## ARIMA Components

```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=== ARIMA COMPONENTS ===")
print("""
ARIMA(p, d, q) has three components:

AR (AutoRegressive) - p:
  - Prediction depends on past values
  - Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ
  - p = number of AR terms (lags)

I (Integrated) - d:
  - Differencing to make series stationary
  - d = number of differencing operations
  - Xₜ' = Xₜ - Xₜ₋₁ (first difference)

MA (Moving Average) - q:
  - Prediction depends on past errors
  - Xₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q
  - q = number of MA terms
  - εₜ = forecast error at time t

ARIMA combines all three:
  Xₜ' = c + Σφᵢ Xₜ₋ᵢ' + Σθⱼ εₜ₋ⱼ + εₜ
""")
```

## Creating Sample Data

```python
print("\n=== SAMPLE DATA ===")

# Create time series with known properties
dates = pd.date_range('2020-01-01', periods=500, freq='D')

# AR(1) process: Xₜ = 0.8*Xₜ₋₁ + εₜ
np.random.seed(42)
ar1 = np.zeros(500)
for t in range(1, 500):
    ar1[t] = 0.8 * ar1[t-1] + np.random.randn()

# MA(1) process: Xₜ = εₜ + 0.6*εₜ₋₁
np.random.seed(42)
errors = np.random.randn(500)
ma1 = errors.copy()
for t in range(1, 500):
    ma1[t] = errors[t] + 0.6 * errors[t-1]

# ARMA(1,1) process
np.random.seed(42)
errors = np.random.randn(500)
arma11 = np.zeros(500)
for t in range(1, 500):
    arma11[t] = 0.7 * arma11[t-1] + errors[t] + 0.4 * errors[t-1]

ts_ar1 = pd.Series(ar1, index=dates, name='AR(1)')
ts_ma1 = pd.Series(ma1, index=dates, name='MA(1)')
ts_arma = pd.Series(arma11, index=dates, name='ARMA(1,1)')

print("Created sample time series:")
print(f"  AR(1): mean={ts_ar1.mean():.2f}, std={ts_ar1.std():.2f}")
print(f"  MA(1): mean={ts_ma1.mean():.2f}, std={ts_ma1.std():.2f}")
print(f"  ARMA(1,1): mean={ts_arma.mean():.2f}, std={ts_arma.std():.2f}")
```

## Stationarity Testing

```python
print("\n=== STATIONARITY TESTING ===")
print("""
ARIMA requires stationary data (after differencing).

Augmented Dickey-Fuller (ADF) Test:
  H₀: Series has unit root (non-stationary)
  H₁: Series is stationary
  
  If p-value < 0.05: Reject H₀, series is stationary
""")

from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    """Perform ADF test and print results"""
    result = adfuller(series.dropna())
    print(f"ADF Test for {name}:")
    print(f"  Test statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Stationary: {'Yes' if result[1] < 0.05 else 'No'}")
    return result[1] < 0.05

# Test our series
adf_test(ts_ar1, 'AR(1)')
print()

# Create non-stationary series (random walk)
rw = np.cumsum(np.random.randn(500))
ts_rw = pd.Series(rw, index=dates)
adf_test(ts_rw, 'Random Walk')
print()

# After differencing
ts_rw_diff = ts_rw.diff().dropna()
adf_test(ts_rw_diff, 'Random Walk (differenced)')
```

## ACF and PACF for Model Selection

```python
print("\n=== ACF AND PACF ===")
print("""
Use ACF and PACF plots to identify p and q:

ACF (Autocorrelation Function):
  - AR(p): Decays gradually
  - MA(q): Cuts off after lag q

PACF (Partial Autocorrelation Function):
  - AR(p): Cuts off after lag p
  - MA(q): Decays gradually

IDENTIFICATION RULES:
  Pattern       |  ACF              |  PACF
  AR(p)         |  Tails off        |  Cuts off at lag p
  MA(q)         |  Cuts off at q    |  Tails off
  ARMA(p,q)     |  Tails off        |  Tails off
""")

from statsmodels.tsa.stattools import acf, pacf

# Calculate ACF and PACF for AR(1)
acf_vals = acf(ts_ar1, nlags=20)
pacf_vals = pacf(ts_ar1, nlags=20)

print("AR(1) process - ACF and PACF:")
print(f"{'Lag':>4} | {'ACF':>8} | {'PACF':>8}")
print("-" * 25)
for lag in range(11):
    print(f"{lag:>4} | {acf_vals[lag]:>8.3f} | {pacf_vals[lag]:>8.3f}")

print("""
Note: PACF cuts off after lag 1 (significant only at lag 1)
      ACF decays gradually
      → AR(1) pattern confirmed
""")
```

## Fitting ARIMA Models

```python
print("\n=== FITTING ARIMA MODELS ===")

from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(1,0,0) to AR(1) data
model_ar = ARIMA(ts_ar1, order=(1, 0, 0))
result_ar = model_ar.fit()

print("ARIMA(1,0,0) fitted to AR(1) data:")
print(f"  Estimated AR coefficient: {result_ar.params['ar.L1']:.3f} (true: 0.80)")
print(f"  AIC: {result_ar.aic:.1f}")
print(f"  BIC: {result_ar.bic:.1f}")

# Fit ARIMA(0,0,1) to MA(1) data
model_ma = ARIMA(ts_ma1, order=(0, 0, 1))
result_ma = model_ma.fit()

print("\nARIMA(0,0,1) fitted to MA(1) data:")
print(f"  Estimated MA coefficient: {result_ma.params['ma.L1']:.3f} (true: 0.60)")
print(f"  AIC: {result_ma.aic:.1f}")

# Fit ARIMA(1,0,1) to ARMA(1,1) data
model_arma = ARIMA(ts_arma, order=(1, 0, 1))
result_arma = model_arma.fit()

print("\nARIMA(1,0,1) fitted to ARMA(1,1) data:")
print(f"  AR coefficient: {result_arma.params['ar.L1']:.3f} (true: 0.70)")
print(f"  MA coefficient: {result_arma.params['ma.L1']:.3f} (true: 0.40)")
```

## Model Selection with AIC/BIC

```python
print("\n=== MODEL SELECTION ===")
print("""
AIC (Akaike Information Criterion):
  AIC = -2*log(L) + 2*k
  - Penalizes complexity less
  
BIC (Bayesian Information Criterion):
  BIC = -2*log(L) + log(n)*k
  - Stronger penalty for complexity

Lower values are better.
Compare models on same data.
""")

# Grid search for best model
print("Model selection for AR(1) data:")
print(f"{'Model':>12} | {'AIC':>10} | {'BIC':>10}")
print("-" * 40)

best_aic = float('inf')
best_order = None

for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(ts_ar1, order=(p, 0, q))
            result = model.fit()
            print(f"ARIMA({p},0,{q}) | {result.aic:>10.1f} | {result.bic:>10.1f}")
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, 0, q)
        except:
            pass

print(f"\nBest model by AIC: ARIMA{best_order}")
```

## Auto ARIMA

```python
print("\n=== AUTO ARIMA ===")
print("""
pmdarima.auto_arima automatically selects best model:
  - Tests stationarity
  - Determines d
  - Searches over p and q
  - Uses AIC/BIC for selection

Parameters:
  - seasonal: Include seasonal component
  - m: Seasonal period
  - stepwise: Faster but may miss optimal
""")

# Simplified auto selection
def simple_auto_arima(series, max_p=3, max_q=3):
    """Simple auto ARIMA selection"""
    best_aic = float('inf')
    best_order = (0, 0, 0)
    best_model = None
    
    # Check stationarity
    d = 0
    test_series = series.copy()
    while adfuller(test_series.dropna())[1] > 0.05 and d < 2:
        test_series = test_series.diff().dropna()
        d += 1
    
    # Grid search
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, d, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
                    best_model = result
            except:
                pass
    
    return best_order, best_model

order, model = simple_auto_arima(ts_arma)
print(f"Auto-selected model: ARIMA{order}")
print(f"AIC: {model.aic:.1f}")
```

## Forecasting

```python
print("\n=== FORECASTING ===")

# Fit model on training data
train = ts_ar1[:400]
test = ts_ar1[400:]

model = ARIMA(train, order=(1, 0, 0))
result = model.fit()

# Forecast
forecast = result.forecast(steps=len(test))
conf_int = result.get_forecast(steps=len(test)).conf_int()

print("Forecast comparison (first 10 steps):")
print(f"{'Step':>5} | {'Forecast':>10} | {'Actual':>10} | {'Error':>10}")
print("-" * 45)

errors = []
for i in range(10):
    fc = forecast.iloc[i]
    ac = test.iloc[i]
    err = fc - ac
    errors.append(err)
    print(f"{i+1:>5} | {fc:>10.3f} | {ac:>10.3f} | {err:>10.3f}")

# Forecast accuracy
mae = np.abs(forecast - test).mean()
rmse = np.sqrt(((forecast - test) ** 2).mean())
print(f"\nForecast accuracy:")
print(f"  MAE: {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
```

## Residual Diagnostics

```python
print("\n=== RESIDUAL DIAGNOSTICS ===")
print("""
Good model should have residuals that are:
  1. Approximately normally distributed
  2. No autocorrelation (white noise)
  3. Constant variance (homoscedastic)

Check:
  - ACF of residuals (should be near 0)
  - Ljung-Box test (no significant autocorrelation)
""")

# Get residuals
residuals = result.resid

print("Residual statistics:")
print(f"  Mean: {residuals.mean():.4f} (should be ~0)")
print(f"  Std: {residuals.std():.4f}")

# ACF of residuals
resid_acf = acf(residuals, nlags=10)
print(f"\nResidual ACF (first 5 lags):")
for lag in range(1, 6):
    print(f"  Lag {lag}: {resid_acf[lag]:.3f}")

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10])
print(f"\nLjung-Box test (lag 10):")
print(f"  Statistic: {lb_test['lb_stat'].iloc[0]:.2f}")
print(f"  p-value: {lb_test['lb_pvalue'].iloc[0]:.3f}")
print(f"  No autocorrelation: {'Yes' if lb_test['lb_pvalue'].iloc[0] > 0.05 else 'No'}")
```

## Key Points

- **ARIMA(p,d,q)**: Combines AR, differencing, and MA
- **p**: Number of autoregressive lags
- **d**: Number of differences for stationarity
- **q**: Number of moving average lags
- **Stationarity**: Required, use ADF test to check
- **ACF/PACF**: Help identify p and q parameters
- **AIC/BIC**: Model selection criteria (lower is better)
- **Residuals**: Should be white noise (no autocorrelation)

## Reflection Questions

1. How does the 'd' parameter in ARIMA relate to trend in the data?
2. Why might you choose a simpler model even if a complex one has slightly better AIC?
3. What would residual autocorrelation indicate about your ARIMA model?
