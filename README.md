# Keras Reservoir Computing

KRC provides Echo State Networks and related reservoir computing utilities built on TensorFlow/Keras. It focuses on fast training of readout layers and convenient tools for forecasting and hyper‑parameter optimisation.

## Installation

### Using Conda

```bash
conda create --name krc python=3.12
conda activate krc
pip install .
```

### Using venv/pip

```bash
python -m venv krc
source krc/bin/activate  # on Windows: krc\Scripts\activate
pip install .
```

## Quick Start

### Instantiating a Model

Pre‑built architectures are available from `keras_reservoir_computing.models.architectures`:

```python
from keras_reservoir_computing.models.architectures import classic_ESN

model = classic_ESN(
    units=200,
    batch=1,
    features=1,
)
```

Models can also be assembled using the Keras Functional API and the provided builders:

```python
import tensorflow as tf
from keras_reservoir_computing.layers.builders import ESNReservoir_builder, ReadOut_builder

inputs = tf.keras.layers.Input(shape=(None, 1), batch_size=1)
reservoir = ESNReservoir_builder({"units": 200})(inputs)
outputs = ReadOut_builder({"kind": "ridge", "units": 1})(reservoir)
model = tf.keras.Model(inputs, outputs)
```

### Training

Reservoir layers are left untouched during training. Only the readout layers are fitted using `ReservoirTrainer`:

```python
import keras_reservoir_computing as krc

trainer = krc.training.ReservoirTrainer(
    model,
    readout_targets={"readout": y_train},
    log=True,
)
trainer.fit_readout_layers(warmup_data=X_train[:,:100], input_data=X_train)
```

### Forecasting

After training, forecasts are generated auto‑regressively. `warmup_forecast` runs a warm‑up pass on real data before forecasting, while `forecast` starts directly from an initial feedback vector.

The optional `states` argument controls whether hidden reservoir states are returned in addition to the predictions.

```python
preds, states = krc.forecasting.warmup_forecast(
    model,
    warmup_data=X_val,
    forecast_data=X_seed,
    horizon=500,
    states=True,      # set False to skip state history
)
```

### Hyper‑parameter Optimisation

Basic hyper‑parameter optimisation is handled by `run_hpo`. A full example is provided in [docs/hyperparameter_optimization.md](docs/hyperparameter_optimization.md), but the general workflow is:

```python
from keras_reservoir_computing.hpo import run_hpo

study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=50,
    trainer="custom",
    loss="efh",
    loss_params={"metric": "rmse", "threshold": 0.2, "softness": 0.02},
    study_name="hpo_example",
    storage="sqlite:///hpo_example.db",
)
```

`study.best_params` holds the best parameters and can be used to instantiate the final model.

---

For advanced configuration options and additional examples see [docs/advanced_usage.md](docs/advanced_usage.md).

