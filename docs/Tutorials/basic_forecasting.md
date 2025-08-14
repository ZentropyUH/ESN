# Tutorial: Basic time-series forecasting

This walkthrough trains a classic ESN to predict a univariate time series, then forecasts 500 steps ahead.

## Setup

```python
import numpy as np
import tensorflow as tf
import keras_reservoir_computing as krc
from keras_reservoir_computing.models import classic_ESN
from keras_reservoir_computing.training import ReservoirTrainer
```

## Data

Create a synthetic signal for demonstration:

```python
T = 12000
train_T = 8000
val_T = 2000
horizon = 500

x = np.sin(np.linspace(0, 120 * np.pi, T))
X = x.reshape(1, T, 1).astype("float32")

X_train = X[:, :train_T, :]
y_train = X[:, 1:train_T + 1, :]
X_val   = X[:, train_T:train_T + val_T, :]
X_seed  = X[:, train_T + val_T - 1: train_T + val_T, :]
```

## Model

```python
model = classic_ESN(
    units=300,
    batch=1,
    features=1,
    dtype="float32",
)
```

## Train readout

```python
trainer = ReservoirTrainer(
    model,
    readout_targets={"ReadOut": y_train},  # default readout name in defaults
    log=True,
)
trainer.fit_readout_layers(warmup_data=X_train[:, :200], input_data=X_train)
```

If your readout was named explicitly (e.g., `name="readout"`), use that key instead.

## Forecast

```python
preds, states = krc.forecasting.warmup_forecast(
    model,
    warmup_data=X_val,
    forecast_data=X_seed,
    horizon=horizon,
    states=True,
)
print(preds.shape)
```

## Next steps

- Try `Ott_ESN` for state augmentation.
- Tune `units`, `leak_rate`, and `spectral_radius` (via kernel initializer) for longer memory.