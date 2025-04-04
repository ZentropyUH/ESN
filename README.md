# Keras Reservoir Computing (KRC)

A TensorFlow/Keras implementation of reservoir computing with a focus on Echo State Networks (ESNs). This library provides a flexible framework for creating, training, and evaluating reservoir computing models integrated with the Keras API.

## Installation

### Using Conda

Create a conda environment with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate krc
```

### Using pip

Create a virtual environment with the `requirements.txt` file:

```bash
python -m venv krc
source krc/bin/activate  # On Windows: krc\Scripts\activate
pip install -r requirements.txt
```

### Install the package

Install the package from the repository root:

```bash
pip install .
```

## Keras Reservoir Computing Workflow

This guide demonstrates how to implement, train, and use reservoir computing models with the KRC library. Reservoir computing models are particularly suitable for time series forecasting.

### 1. Using Pre-built Model Architectures

KRC provides several pre-built model architectures for common reservoir computing tasks:

```python
import tensorflow as tf
import numpy as np
import keras_reservoir_computing as krc
from keras_reservoir_computing.models.architectures import (
    classic_ESN,
    Ott_ESN,
    ensemble_with_mean_ESN,
    residual_stacked_ESN
)
```

#### Classic Echo State Network

```python
# Create a basic ESN model
model = classic_ESN(
    units=300,            # Reservoir size
    batch=1,              # Batch size
    features=3            # Dimensionality of input/output data
)
```

#### Ott's ESN Model with State Augmentation

```python
# Create an ESN model with Edward Ott's reservoir architecture
# This model augments the reservoir output with squared values and concatenates with input
model = Ott_ESN(
    units=300,
    batch=1,
    features=3
)
```

#### Ensemble of ESNs with Mean Aggregation

```python
# Create an ensemble of ESN models
model = ensemble_with_mean_ESN(
    units=300,
    ensemble_size=5,     # Number of reservoir models in the ensemble
    batch=1,
    features=3
)
```

#### Residual Stacked ESN

```python
# Create a multi-layer ESN with residual connections
model = residual_stacked_ESN(
    units=[300, 200, 100],  # Units per reservoir layer
    batch=1,
    features=3
)
```

### 2. Configuring Reservoirs and Readouts with JSON

KRC makes it easy to configure models using JSON/dictionary configurations:

```python
# Custom reservoir configuration
reservoir_config = {
    "units": 300,
    "feedback_dim": 3,          # Dimension of feedback
    "input_dim": 0,             # Set to 0 for feedback-only
    "leak_rate": 0.6,           # Controls memory of the reservoir
    "activation": "tanh",
    "kernel_initializer": {
        "name": "WattsStrogatzGraphInitializer",
        "params": {
            "k": 10,            # Number of neighbors in small-world graph
            "p": 0.1,           # Rewiring probability
            "spectral_radius": 0.9,  # Controls echo state property
            "directed": True,
            "seed": 42
        }
    },
    "feedback_initializer": {
        "name": "PseudoDiagonalInitializer",
        "params": {"sigma": 0.5, "binarize": False, "seed": 42}
    }
}

# Custom readout configuration
readout_config = {
    "kind": "ridge",            # Use ridge regression ("ridge" or "mpenrose")
    "units": 3,                 # Output dimension
    "alpha": 1e-6,              # Regularization strength
    "trainable": False          # Not trained with gradient descent
}

# Create model with custom configuration
model = classic_ESN(
    units=300,
    reservoir_config=reservoir_config,
    readout_config=readout_config,
    batch=1,
    features=3
)
```

### 3. Building Custom Reservoir Models

For more flexibility, you can create custom reservoir models using the builder functions:

```python
from keras_reservoir_computing.layers.builders import ESNReservoir_builder, ReadOut_builder

# Create input layer
inputs = tf.keras.layers.Input(shape=(None, 3), batch_size=1)

# Build reservoir using configuration dictionary
reservoir = ESNReservoir_builder(reservoir_config)(inputs)

# Optional: Add additional processing to reservoir outputs
processed = tf.keras.layers.Dense(50, activation='relu')(reservoir)

# Build readout layer
outputs = ReadOut_builder(readout_config)(processed)

# Create model
custom_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
```

### 4. Training Reservoir Computing Models

Unlike traditional neural networks, reservoir computing models are not trained with gradient descent. KRC provides a custom trainer for readout layers:

```python
from keras_reservoir_computing.training.training import ReservoirTrainer

# Prepare your data
# X_train: shape (batch_size, timesteps, features)
# y_train: shape (batch_size, timesteps, targets)

# Create a trainer for the model
trainer = ReservoirTrainer(
    model=model,
    readout_targets={"ridge_svdreadout": y_train}  # Map readout layers to targets
)

# Train the model
# First run a warm-up phase to initialize the reservoir state
warmup_data = X_train[:, :100, :]  # Use first 100 timesteps as warmup

# Fit all readout layers in the correct order
trainer.fit_readout_layers(
    warmup_data=warmup_data,
    X=X_train
)
```

#### Key Training Implementation Details

- **Reservoir State Preservation**: The warmup phase is crucial as it initializes the internal states of reservoir layers before training begins. These states are maintained throughout the training process.

- **ReadOut Training Process**:
  1. For each ReadOut layer in topological order:
     - An intermediate model is created to extract inputs for that layer
     - The warmup phase runs to initialize reservoir states
     - Inputs are extracted and passed to the ReadOut layer's fit method
  
- **Ridge Regression with SVD**:
  - ReadOut layers like RidgeSVDReadout use ridge regression with SVD decomposition for numerical stability
  - The fit method automatically:
    - Casts inputs to float64 for numerical precision
    - Flattens 3D inputs (batch, timesteps, features) to 2D (samples, features)
    - Centers the data before computing the SVD
    - Applies regularization via the alpha parameter
    - Computes weights analytically and assigns them to the layer

### 5. Generating Forecasts

Once trained, the model can be used for generating multi-step forecasts:

```python
from keras_reservoir_computing.forecasting.forecasting import forecast, warmup_forecast

# For generative forecasting (predicting future steps beyond training data)
forecast_horizon = 1000  # Number of steps to forecast

# Option 1: Basic forecast (starting from the last training point)
initial_feedback = X_train[:, -1:, :]  # Use last timestep as initial condition
external_inputs = ()  # No external inputs for a feedback-only model

# Generate forecast
predictions, states = forecast(
    model=model,
    initial_feedback=initial_feedback,
    horizon=forecast_horizon,
    external_inputs=external_inputs
)

# Option 2: Warmup forecast (runs a warmup phase on actual data before forecasting)
warmup_data = X_train[:, -200:, :]  # Use last 200 timesteps for warmup
forecast_data = X_test  # The test data we want to forecast

predictions, states = warmup_forecast(
    model=model,
    warmup_data=warmup_data,
    forecast_data=forecast_data,
    horizon=forecast_horizon
)
```

#### Key Forecasting Implementation Details

- **Auto-regressive Process**: The forecasting is auto-regressive, meaning each prediction is fed back as input for generating the next step.

- **Reservoir State Tracking**: The forecasting functions maintain and track the internal states of all reservoir layers during prediction, returning these states along with the predictions.

- **Warmup Importance**: The `warmup_forecast` function is generally preferred as it:
  1. Properly initializes reservoir states by running the model on actual data
  2. Creates a smooth transition between observed data and forecasts
  3. Reduces initial forecast errors that can occur with randomly initialized states

- **Efficient Implementation**: The forecasting functions use TensorFlow's `tf.while_loop` with shape invariants for efficient execution, making it possible to generate long forecasts without excessive memory usage.

- **External Input Support**: For models that use external inputs beyond feedback, you can provide these as time series to incorporate known future information during forecasting.

### 6. Integrating Reservoir Models into Larger Keras Networks

You can incorporate reservoir layers into larger Keras architectures:

```python
# Create a more complex model with reservoir and standard Keras layers

# Multiple inputs
feedback_input = tf.keras.layers.Input(shape=(None, 3), name="feedback_input")
external_input = tf.keras.layers.Input(shape=(None, 5), name="external_input")

# Pre-process external inputs
processed_inputs = tf.keras.layers.Dense(10, activation='relu')(external_input)

# Combine inputs
combined = tf.keras.layers.Concatenate()([feedback_input, processed_inputs])

# Create reservoir with external inputs
reservoir_config["input_dim"] = 10  # Match dimension of combined inputs
reservoir = ESNReservoir_builder(reservoir_config)(combined)

# Add downstream layers
x = tf.keras.layers.Dense(50, activation='relu')(reservoir)
outputs = ReadOut_builder(readout_config)(x)

# Create hybrid model with multiple inputs
hybrid_model = tf.keras.models.Model(
    inputs=[feedback_input, external_input],
    outputs=outputs
)

# Training works the same way with the ReservoirTrainer
```

## Advanced Usage

### Graph-Based Reservoir Initializers

The library provides several graph-based initializers for reservoir weights:

```python
from keras_reservoir_computing.initializers.recurrent_initializers.graph_initializers import (
    WattsStrogatzGraphInitializer,    # Small-world networks
    ErdosRenyiGraphInitializer,       # Random graphs
    BarabasiAlbertGraphInitializer,   # Scale-free networks
    NewmanWattsStrogatzGraphInitializer,
    RegularGraphInitializer,
    CompleteGraphInitializer
)
```

Each initializer creates reservoirs with different topological properties:

```python
# Small-world topology (good for most tasks)
small_world_init = WattsStrogatzGraphInitializer(
    k=10,                  # Number of neighbors
    p=0.1,                 # Rewiring probability
    spectral_radius=0.9,   # Controls stability
    directed=True          # Directed connections
)

# Scale-free network (higher connectivity variation)
scale_free_init = BarabasiAlbertGraphInitializer(
    m=3,                   # New nodes form m connections
    spectral_radius=0.9,
    directed=True
)

# Random graph topology
random_init = ErdosRenyiGraphInitializer(
    p=0.1,                 # Connection probability
    spectral_radius=0.9,
    directed=True
)
```

### Readout Methods

The library supports different readout training approaches:

```python
# Ridge regression with SVD (better numerical stability)
ridge_config = {
    "kind": "ridge",
    "units": 3,
    "alpha": 1e-5          # Regularization strength
}

# Moore-Penrose pseudoinverse method
mpenrose_config = {
    "kind": "mpenrose",
    "units": 3
}
```

## Saving and Loading Models

Reservoir models can be saved and loaded like any Keras model:

```python
# Save the model
model.save('my_reservoir_model')

# Load the model
loaded_model = tf.keras.models.load_model('my_reservoir_model')
```

## Examples

See the `/examples` directory for complete examples of:
- Time series forecasting with ESNs
- Working with multiple inputs
- Ensemble models
- Hyperparameter tuning