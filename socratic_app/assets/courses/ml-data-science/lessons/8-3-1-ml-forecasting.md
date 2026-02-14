# Time Series Forecasting with Machine Learning

## Introduction

Beyond traditional statistical methods, machine learning approaches can capture complex patterns in time series data. This lesson covers feature engineering and ML models for time series forecasting.

## Feature Engineering for Time Series

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

print("=== ML FOR TIME SERIES ===")
print("""
Machine learning can capture:
  - Complex non-linear patterns
  - Multiple seasonalities
  - External features (weather, holidays)
  - Interactions between features

Key difference from traditional ML:
  - Must preserve temporal order
  - Create lag features (past values)
  - Time-based train/test splits
""")

# Create sample time series
dates = pd.date_range('2022-01-01', periods=730, freq='D')
t = np.arange(730)

# Complex pattern
trend = 50 + 0.05 * t
yearly_seasonal = 15 * np.sin(2 * np.pi * t / 365.25)
weekly_seasonal = 5 * np.sin(2 * np.pi * t / 7)
noise = np.random.randn(730) * 5

y = trend + yearly_seasonal + weekly_seasonal + noise

df = pd.DataFrame({'date': dates, 'value': y})
df = df.set_index('date')

print(f"Created time series: {len(df)} days")
print(f"Range: [{df['value'].min():.1f}, {df['value'].max():.1f}]")
```

## Creating Lag Features

```python
print("\n=== LAG FEATURES ===")
print("""
Lag features = past values of the target variable

y_lag1 = y(t-1)   (yesterday's value)
y_lag7 = y(t-7)   (same day last week)
y_lag365 = y(t-365) (same day last year)

Also called "autoregressive features"
""")

def create_lag_features(df, lags):
    """Create lag features"""
    result = df.copy()
    for lag in lags:
        result[f'lag_{lag}'] = result['value'].shift(lag)
    return result

# Create lag features
lags = [1, 2, 3, 7, 14, 30]
df_features = create_lag_features(df, lags)

print("Lag features (sample):")
print(df_features.iloc[35:40].round(2))
```

## Time-Based Features

```python
print("\n=== TIME-BASED FEATURES ===")
print("""
Extract features from the timestamp:
  - Day of week (0-6)
  - Day of month (1-31)
  - Month (1-12)
  - Quarter (1-4)
  - Year
  - Is weekend
  - Is holiday
""")

def create_time_features(df):
    """Create time-based features"""
    result = df.copy()
    result['dayofweek'] = result.index.dayofweek
    result['dayofmonth'] = result.index.day
    result['month'] = result.index.month
    result['quarter'] = result.index.quarter
    result['year'] = result.index.year
    result['is_weekend'] = (result.index.dayofweek >= 5).astype(int)
    result['dayofyear'] = result.index.dayofyear
    return result

df_features = create_time_features(df_features)

print("Time-based features (sample):")
print(df_features[['value', 'dayofweek', 'month', 'is_weekend', 'dayofyear']].iloc[0:5])
```

## Rolling Statistics

```python
print("\n=== ROLLING STATISTICS ===")
print("""
Rolling/moving statistics capture recent trends:
  - Rolling mean (smoothed level)
  - Rolling std (recent volatility)
  - Rolling min/max (recent range)
""")

def create_rolling_features(df, windows):
    """Create rolling statistics"""
    result = df.copy()
    for window in windows:
        result[f'rolling_mean_{window}'] = result['value'].shift(1).rolling(window).mean()
        result[f'rolling_std_{window}'] = result['value'].shift(1).rolling(window).std()
    return result

windows = [7, 14, 30]
df_features = create_rolling_features(df_features, windows)

print("Rolling features (sample):")
cols = ['value', 'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']
print(df_features[cols].iloc[35:40].round(2))
```

## Preparing Data for ML

```python
print("\n=== PREPARING DATA ===")

# Remove rows with NaN (from lags and rolling)
df_ml = df_features.dropna()

print(f"Samples after removing NaN: {len(df_ml)}")

# Define features and target
feature_cols = [col for col in df_ml.columns if col != 'value']
X = df_ml[feature_cols]
y = df_ml['value']

print(f"\nFeatures ({len(feature_cols)}):")
print(feature_cols)

# Time-based split (no shuffling!)
train_size = int(len(df_ml) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nTrain: {X_train.index[0]} to {X_train.index[-1]} ({len(X_train)} samples)")
print(f"Test: {X_test.index[0]} to {X_test.index[-1]} ({len(X_test)} samples)")
```

## Training ML Models

```python
print("\n=== TRAINING MODELS ===")

models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse
    })
    print(f"\n{name}:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")

# Compare with naive forecast (yesterday's value)
naive_pred = df_ml['lag_1'].iloc[train_size:].values
naive_mae = mean_absolute_error(y_test, naive_pred)
print(f"\nNaive (lag-1) MAE: {naive_mae:.3f}")
```

## Feature Importance

```python
print("\n=== FEATURE IMPORTANCE ===")

# Random Forest feature importance
rf_model = models['Random Forest']
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features (Random Forest):")
print(importances.head(10).to_string(index=False))

print("""
Observations:
  - Lag features (especially recent) are most important
  - Rolling statistics capture trends
  - Time features capture seasonality
""")
```

## Cross-Validation for Time Series

```python
print("\n=== TIME SERIES CROSS-VALIDATION ===")
print("""
Cannot use random k-fold for time series!

TimeSeriesSplit:
  - Always train on past, test on future
  - Expanding training window

Example (5 splits):
  Split 1: Train [===   ] Test [ = ]
  Split 2: Train [====  ] Test [  =]
  Split 3: Train [===== ] Test [   =]
  ...
""")

tscv = TimeSeriesSplit(n_splits=5)

print("Cross-validation scores (Random Forest):")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

cv_scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    cv_scores.append(mae)
    
    print(f"  Fold {fold+1}: MAE = {mae:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")

print(f"\nMean CV MAE: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

## Multi-Step Forecasting

```python
print("\n=== MULTI-STEP FORECASTING ===")
print("""
Two approaches for predicting multiple steps ahead:

1. RECURSIVE (iterative):
   - Predict one step
   - Use prediction as feature for next step
   - Errors can accumulate

2. DIRECT:
   - Train separate model for each horizon
   - More models but no error accumulation

3. MULTI-OUTPUT:
   - Single model predicts all horizons
   - Complex but can capture dependencies
""")

def recursive_forecast(model, X_last, steps, feature_cols, lags):
    """Recursive multi-step forecast"""
    forecasts = []
    X_current = X_last.copy()
    
    for step in range(steps):
        # Predict
        pred = model.predict(X_current.values.reshape(1, -1))[0]
        forecasts.append(pred)
        
        # Update features for next step
        # Shift lags
        for i in range(len(lags)-1, 0, -1):
            lag_col = f'lag_{lags[i]}'
            prev_lag_col = f'lag_{lags[i-1]}'
            if lag_col in feature_cols and prev_lag_col in feature_cols:
                X_current[lag_col] = X_current[prev_lag_col]
        
        X_current['lag_1'] = pred
        
    return np.array(forecasts)

# Demonstrate recursive forecast
rf_model = models['Random Forest']
X_last = X_test.iloc[-1]

print("5-step recursive forecast:")
forecasts = recursive_forecast(rf_model, X_last, 5, feature_cols, lags)
print(f"  {forecasts.round(2)}")
```

## Key Points

- **Feature engineering**: Lags, time features, rolling statistics
- **Temporal order**: Always preserve, no shuffling
- **Time-based split**: Train on past, test on future
- **TimeSeriesSplit**: Proper cross-validation for time series
- **Multiple models**: RF, GBM often work well
- **Lag features**: Usually most important
- **Multi-step**: Recursive or direct approaches

## Reflection Questions

1. Why are lag features typically the most important in time series ML models?
2. What problems could arise from using random train/test splits for time series?
3. When might traditional statistical methods (ARIMA) outperform ML approaches?
