# Advanced Usage

This document collects more advanced topics that go beyond the quick start in the main README. It shows how to fully configure reservoirs, integrate them with other Keras components and customize the internal graph initializers.

## Configuring Reservoirs and Readouts with JSON

KRC models can be built from configuration dictionaries (or JSON files). The `classic_ESN` helper accepts `reservoir_config` and `readout_config` dictionaries:

```python
reservoir_config = {
                "class_name": "krc>ESNReservoir",
                "config": {
                    "units": 100,
                    "feedback_dim": 1,
                    "input_dim": 0,
                    "leak_rate": 1.0,
                    "activation": "tanh",
                    "input_initializer": {
                    "class_name": "zeros",
                    "config": {}
                    },
                    "feedback_initializer": {
                    "class_name": "krc>RandomInputInitializer",
                    "config": {
                        "input_scaling": 1,
                        "seed": 42
                    }
                    },
                    "feedback_bias_initializer": {
                    "class_name": "zeros",
                    "config": {}
                    },
                    "kernel_initializer": {
                    "class_name": "krc>RandomRecurrentInitializer",
                    "config": {
                        "density": 0.01,
                        "spectral_radius": 0.9,
                        "seed": 42
                    }
                    },
                    "dtype": "float32"
                }
            }

readout_config = {
                "class_name": "krc>RidgeReadout",
                "config": {
                    "units": 1,
                    "alpha": 0.1,
                    "max_iter": 1000,
                    "tol": 0.000001,
                    "trainable": False,
                    "dtype": "float64"
                }
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
from keras_reservoir_computing.io.loaders import load_object
import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(None, 3), batch_size=1)
reservoir = load_object(reservoir_config)(inputs)
processed = tf.keras.layers.Dense(50, activation="relu")(reservoir)
outputs = load_object(readout_config)(processed)

custom_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
```

Reservoir and readout configs can also be loaded from a file:

```python
reservoir_config = load_config("path/to/reservoir.json")
readout_config = load_config("path/to/readout.json")
```

or from a default config:

```python
reservoir_config = load_default_config("reservoir")
readout_config = load_default_config("readout")
```

Reservoir and readout default configs are stored in `keras_reservoir_computing/io/defaults/`.

## Integrating Reservoirs into Larger Networks

Reservoir layers can be combined with other Keras layers and multiple inputs:

```python
feedback_input = tf.keras.layers.Input(shape=(None, 3), name="feedback_input")
external_input = tf.keras.layers.Input(shape=(None, 5), name="external_input")
processed_inputs = tf.keras.layers.Dense(10, activation="relu")(external_input)
combined = tf.keras.layers.Concatenate()([feedback_input, processed_inputs])
reservoir_config["input_dim"] = 10
reservoir = load_object(reservoir_config)(combined)
x = tf.keras.layers.Dense(50, activation="relu")(reservoir)
outputs = load_object(readout_config)(x)

hybrid_model = tf.keras.models.Model(
    inputs=[feedback_input, external_input],
    outputs=outputs,
)
```

Here the feedback and external inputs are only semantically different, but they are simply two different inputs to the same model. The difference is that in the generative forecast, the feedback is the generative model's output, while the external input is provided by the user.

See this as an input-driven dynamical system.



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
model.save("path/to/my_reservoir_model.keras")
loaded_model = tf.keras.models.load_model("path/to/my_reservoir_model.keras")
```

## Examples

See the `examples/` directory for end-to-end scripts covering time series forecasting, working with multiple inputs, ensemble models and hyperparameter tuning.

