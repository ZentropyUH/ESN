# Advanced Usage

This document collects more advanced topics that go beyond the quick start in the main README. It shows how to fully configure reservoirs, integrate them with other Keras components and customize the internal graph initializers.

## Configuring Reservoirs and Readouts with JSON

KRC models can be built from configuration dictionaries (or JSON files). The `classic_ESN` helper accepts `reservoir_config` and `readout_config` dictionaries:

```python
reservoir_config = {
    "units": 300,
    "feedback_dim": 3,
    "input_dim": 0,
    "leak_rate": 0.6,
    "activation": "tanh",
    "kernel_initializer": {
        "name": "WattsStrogatzGraphInitializer",
        "params": {
            "k": 10,
            "p": 0.1,
            "spectral_radius": 0.9,
            "directed": True,
            "seed": 42,
        },
    },
    "feedback_initializer": {
        "name": "PseudoDiagonalInitializer",
        "params": {"sigma": 0.5, "binarize": False, "seed": 42},
    },
}

readout_config = {
    "kind": "ridge",
    "units": 3,
    "alpha": 1e-6,
    "trainable": False,
}

model = classic_ESN(
    units=300,
    reservoir_config=reservoir_config,
    readout_config=readout_config,
    batch=1,
    features=3,
)
```

## Building Custom Reservoir Models with the Functional API

For full flexibility you can assemble models manually using the provided builders:

```python
from keras_reservoir_computing.layers.builders import ESNReservoir_builder, ReadOut_builder
import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(None, 3), batch_size=1)
reservoir = ESNReservoir_builder(reservoir_config)(inputs)
processed = tf.keras.layers.Dense(50, activation="relu")(reservoir)
outputs = ReadOut_builder(readout_config)(processed)

custom_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
```

## Integrating Reservoirs into Larger Networks

Reservoir layers can be combined with other Keras layers and multiple inputs:

```python
feedback_input = tf.keras.layers.Input(shape=(None, 3), name="feedback_input")
external_input = tf.keras.layers.Input(shape=(None, 5), name="external_input")
processed_inputs = tf.keras.layers.Dense(10, activation="relu")(external_input)
combined = tf.keras.layers.Concatenate()([feedback_input, processed_inputs])
reservoir_config["input_dim"] = 10
reservoir = ESNReservoir_builder(reservoir_config)(combined)
x = tf.keras.layers.Dense(50, activation="relu")(reservoir)
outputs = ReadOut_builder(readout_config)(x)

hybrid_model = tf.keras.models.Model(
    inputs=[feedback_input, external_input],
    outputs=outputs,
)
```

## Graph-Based Reservoir Initializers

Several initializers create reservoirs with different connectivity patterns:

```python
from keras_reservoir_computing.initializers.recurrent_initializers.graph_initializers import (
    WattsStrogatzGraphInitializer,
    ErdosRenyiGraphInitializer,
    BarabasiAlbertGraphInitializer,
    NewmanWattsStrogatzGraphInitializer,
    RegularGraphInitializer,
    CompleteGraphInitializer,
)

small_world = WattsStrogatzGraphInitializer(k=10, p=0.1, spectral_radius=0.9, directed=True)
scale_free = BarabasiAlbertGraphInitializer(m=3, spectral_radius=0.9, directed=True)
random_graph = ErdosRenyiGraphInitializer(p=0.1, spectral_radius=0.9, directed=True)
```

## Saving and Loading Models

Reservoir models can be saved and later loaded like any Keras model:

```python
model.save("my_reservoir_model")
loaded_model = tf.keras.models.load_model("my_reservoir_model")
```

## Examples

See the `examples/` directory for end-to-end scripts covering time series forecasting, working with multiple inputs, ensemble models and hyperparameter tuning.

