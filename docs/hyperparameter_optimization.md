# Hyperparameter Optimisation

This example demonstrates how to use the `run_hpo` helper to tune an ESN. The data is stored in a sequence of `.npy` files which are concatenated along the batch dimension.

```python
import keras_reservoir_computing as krc
from keras_reservoir_computing.hpo import run_hpo
import numpy as np

# Paths to data sequences
# Here we load ten Mackey--Glass time series chunks
DATA_PATHS = [f"/media/elessar/Data/Pincha/MachineLearning/ESN/Research/curvature_dt_estimation/Data/MG/MG_{i}.npy" for i in range(1, 11)]

# ----------------------------------------------------------------------
# 1) Model creator
# ----------------------------------------------------------------------

def model_creator(leak_rate: float, input_scaling: float, spectral_radius: float, alpha: float):
    reservoir_conf = {
        "class_name": "krc>ESNReservoir",
        "config": {
            "units": 200,
            "feedback_dim": 1,
            "input_dim": 0,
            "leak_rate": leak_rate,
            "activation": "tanh",
            "input_initializer": {"class_name": "zeros", "config": {}},
            "feedback_initializer": {
                "class_name": "krc>PseudoDiagonalInitializer",
                "config": {"input_scaling": input_scaling, "seed": 7},
            },
            "feedback_bias_initializer": {
                "class_name": "random_uniform",
                "config": {"minval": -0.1, "maxval": 0.1, "seed": 7},
            },
            "kernel_initializer": {
                "class_name": "krc>WattsStrogatzGraphInitializer",
                "config": {
                    "k": 6,
                    "p": 1,
                    "directed": True,
                    "self_loops": True,
                    "tries": 100,
                    "spectral_radius": spectral_radius,
                    "seed": 7,
                },
            },
            "dtype": "float64",
        },
    }

    readout_conf = {
        "class_name": "krc>RidgeReadout",
        "config": {"units": 1, "alpha": alpha, "trainable": False, "name": "readout"},
    }

    from keras_reservoir_computing.models.architectures import classic_ESN

    return classic_ESN(
        units=200,
        reservoir_config=reservoir_conf,
        readout_config=readout_conf,
        batch=10,
        features=1,
        name="ESN",
        dtype="float64",
    )

# ----------------------------------------------------------------------
# 2) Search space
# ----------------------------------------------------------------------

from optuna.trial import Trial

def search_space(trial: Trial):
    leak_rate = trial.suggest_float("leak_rate", 0.1, 1.0)
    min_sr = 1.0 - leak_rate
    spectral_radius = trial.suggest_float("spectral_radius", min_sr, 2.0, log=True)
    return {
        "leak_rate": leak_rate,
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
        "spectral_radius": spectral_radius,
        "alpha": trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
    }

# ----------------------------------------------------------------------
# 3) Data loader
# ----------------------------------------------------------------------

def data_loader(trial=None):
    transient, train, train_target, ftransient, val, val_target = krc.utils.data.load_data(
        datapath=DATA_PATHS,
        init_transient=1000,
        transient_length=2000,
        train_length=10000,
        val_length=5000,
        normalize=True,
    )
    return {
        "transient": transient,
        "train": train,
        "train_target": train_target,
        "ftransient": ftransient,
        "val": val,
        "val_target": val_target,
    }

# ----------------------------------------------------------------------
# 4) Launch the study
# ----------------------------------------------------------------------

study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    n_trials=50,
    data_loader=data_loader,
    loss="efh",
    loss_params={"metric": "rmse", "threshold": 0.2, "softness": 0.02},
    study_name="hpo_example",
    storage="sqlite:///hpo_example.db",
)

best_params = study.best_params
model = model_creator(**best_params)
```
