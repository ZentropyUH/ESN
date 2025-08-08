# Tutorial: Multi-input ESN with exogenous variables

This tutorial shows how to include exogenous inputs alongside the feedback signal.

```python
import tensorflow as tf
from keras_reservoir_computing.io.loaders import load_config, load_object
from keras_reservoir_computing.models import classic_ESN
```

## Build a hybrid model

```python
# Start from default reservoir/readout configs
reservoir_conf = load_config(None)  # default reservoir
readout_conf = load_config(None)    # default readout

# Two inputs: feedback (to be predicted) and external input with 5 features
feedback = tf.keras.layers.Input(shape=(None, 1), batch_size=1, name="feedback")
external = tf.keras.layers.Input(shape=(None, 5), batch_size=1, name="external")

# Combine external features before the reservoir
pre = tf.keras.layers.Dense(10, activation="relu")(external)
concat = tf.keras.layers.Concatenate()([feedback, pre])

# Reservoir expects feedback first; set input_dim accordingly
reservoir_conf.setdefault("config", {})
reservoir_conf["config"].update({"units": 400, "feedback_dim": 1, "input_dim": 10})
reservoir = load_object(reservoir_conf)([feedback, concat])

# Readout over concatenated input + reservoir states
x = tf.keras.layers.Concatenate()([concat, reservoir])
readout = load_object(readout_conf)(x)

hybrid = tf.keras.Model(inputs=[feedback, external], outputs=readout)
```

## Forecasting with exogenous inputs

During rollout, pass a full horizon for the external series. The first forecast step is seeded by the last known feedback:

```python
from keras_reservoir_computing.forecasting import warmup_forecast

preds, states = warmup_forecast(
    hybrid,
    warmup_data=[feedback_warm, external_warm],
    forecast_data=[feedback_seed, external_future],
    horizon=500,
    states=True,
)
```