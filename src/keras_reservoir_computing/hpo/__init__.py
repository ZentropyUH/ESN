# =============================================================
# krc/hpo/__init__.py
# =============================================================
"""Hyper-parameter optimisation (HPO) for reservoir-computing models.

This subpackage provides a production-ready, high-level wrapper around Optuna
for hyperparameter optimization of Keras-based reservoir computing models. It
features robust error handling, efficient memory management, and specialized
loss functions for time series forecasting and chaotic systems.

Public API
----------
- :func:`run_hpo` — Fully managed Optuna study runner with comprehensive validation
- :data:`LOSSES` — Registry of multi-step forecasting losses optimized for RC tasks
- :func:`get_study_summary` — Generate human-readable study summaries

Overview
--------
``run_hpo`` provides a complete HPO pipeline that handles model creation,
training, forecasting, and evaluation. You provide three callbacks:

1. ``model_creator(**params)`` → ``tf.keras.Model``
   
   Creates a fresh, uncompiled model for each trial using the hyperparameters
   from the search space. Must contain at least one ReadOut layer.

2. ``search_space(trial)`` → ``dict[str, Any]``
   
   Defines the hyperparameter search space using ``trial.suggest_*`` methods.
   Returns a dictionary that will be unpacked as keyword arguments to
   ``model_creator``.

3. ``data_loader(trial)`` → ``dict[str, Any]``
   
   Loads and returns training/validation data. Must provide the following keys:
   
   - ``"transient"``: Warmup data for training phase
   - ``"train"``: Training input data
   - ``"train_target"``: Training targets
   - ``"ftransient"``: Warmup data for forecast phase
   - ``"val"``: Validation input data
   - ``"val_target"``: Validation targets
   
   Optionally include ``"readout_targets"`` for multi-readout models.

Training & Evaluation Pipeline
-------------------------------
Each trial follows this workflow:

1. **Model Creation**: ``model = model_creator(**params)``
2. **Readout Training**: Uses :class:`ReservoirTrainer` to fit readout layers
   in topological order with automatic state warmup
3. **Forecasting**: Generates multi-step predictions via :func:`warmup_forecast`
4. **Evaluation**: Computes loss on validation forecasts using the specified
   loss function

Loss Functions
--------------
The :data:`LOSSES` registry contains specialized objectives for reservoir
computing. All operate on batched multi-step forecasts with shape ``(B, T, D)``.

**Built-in losses:**

- ``"efh"`` → :func:`expected_forecast_horizon` **(default, recommended)**
  
  Differentiable proxy for forecast horizon length. Maximizes the expected
  number of time steps where error stays below threshold. Smooth and
  gradient-friendly for chaotic systems.
  
  Parameters: ``threshold`` (default 0.2), ``softness`` (default 0.02)

- ``"lyap"`` → :func:`lyapunov_weighted_loss`
  
  Geometric-mean error weighted by ``exp(-lle * t * dt)``. Emphasizes
  short-term accuracy while accounting for exponential error growth in
  chaotic systems.
  
  Parameters: ``lle`` (Lyapunov exponent), ``dt`` (time step), ``metric``

- ``"horizon"`` → :func:`forecast_horizon_loss`
  
  Negative log of valid forecast horizon. Counts timesteps where geometric
  mean error stays below threshold.
  
  Parameters: ``threshold`` (default 0.2), ``metric`` (default "nrmse")

- ``"multi_step"`` → :func:`multi_step_loss`
  
  Sum of geometric-mean error across all timesteps. General-purpose loss
  for both stable and unstable systems.

- ``"standard"`` → :func:`standard_loss`
  
  Mean of geometric-mean error across timesteps. Simple baseline loss.

Custom Loss Functions
---------------------
You can provide custom loss functions following the ``LossProtocol`` signature:

.. code-block:: python

    def custom_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        /,
        *args,
        **kwargs
    ) -> float:
        \"\"\"Compute loss between true and predicted values.
        
        Parameters
        ----------
        y_true, y_pred : np.ndarray
            Shape (B, T, D) where B=batch, T=time, D=dimensions
            
        Returns
        -------
        float
            Loss value to minimize
        \"\"\"
        # Your loss computation here
        return loss_value

**Requirements:**

- Accept ``y_true`` and ``y_pred`` with shape ``(B, T, D)``
- Return a Python ``float`` (lower is better)
- Should be deterministic and vectorized over batch/time
- Pass extra arguments via ``loss_params`` in :func:`run_hpo`

Study Management
----------------
Studies are automatically saved when ``storage`` is provided (e.g.,
``"sqlite:///study.db"``). The study name defaults to ``<file>:<func>``
via :func:`make_study_name`, enabling easy organization of experiments.

Example
-------
**Basic usage with default EFH loss:**

>>> import tensorflow as tf
>>> from keras_reservoir_computing.hpo import run_hpo, get_study_summary
>>> 
>>> def model_creator(units: int, spectral_radius: float, leak_rate: float):
>>>     # Build and return a reservoir model
>>>     ...
>>>     return model
>>> 
>>> def search_space(trial):
>>>     return {
>>>         "units": trial.suggest_int("units", 200, 1000, step=100),
>>>         "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
>>>         "leak_rate": trial.suggest_float("leak_rate", 0.1, 0.5),
>>>     }
>>> 
>>> def data_loader(trial):
>>>     return {
>>>         "transient": X_trans,
>>>         "train": X_train,
>>>         "train_target": y_train,
>>>         "ftransient": X_ftrans,
>>>         "val": X_val,
>>>         "val_target": y_val,
>>>     }
>>> 
>>> # Run optimization with default EFH loss
>>> study = run_hpo(
>>>     model_creator,
>>>     search_space,
>>>     n_trials=100,
>>>     data_loader=data_loader,
>>>     storage="sqlite:///my_study.db",
>>> )
>>> 
>>> # Print summary
>>> print(get_study_summary(study))
>>> print(f"Best parameters: {study.best_params}")

**Using Lyapunov-weighted loss for chaotic systems:**

>>> study = run_hpo(
>>>     model_creator,
>>>     search_space,
>>>     n_trials=100,
>>>     data_loader=data_loader,
>>>     loss="lyap",
>>>     loss_params={"lle": 1.2, "dt": 0.1},
>>>     storage="sqlite:///chaotic_study.db",
>>> )

Best Practices
--------------
- **Always create fresh models**: Don't reuse or compile models between trials
- **Use EFH loss for chaotic systems**: Provides smooth, differentiable optimization
- **Enable persistent storage**: Use SQLite storage for long-running studies
- **Set appropriate penalty values**: Help constrain invalid parameter combinations
- **Monitor memory usage**: The module handles cleanup automatically, but watch for
  large batch sizes or very deep architectures

Notes
-----
- Models must be fresh and uncompiled for each trial
- Failed trials return ``penalty_value`` (default 1e10) instead of crashing
- The default sampler is :class:`optuna.samplers.TPESampler` with multivariate
  optimization for better handling of parameter dependencies
- Studies automatically resume from the last checkpoint when using persistent storage

See Also
--------
:mod:`keras_reservoir_computing.hpo._losses` : Loss function implementations
:mod:`keras_reservoir_computing.hpo._objective` : Objective function factory
:class:`keras_reservoir_computing.training.ReservoirTrainer` : Readout training
:func:`keras_reservoir_computing.forecasting.warmup_forecast` : Forecasting
"""
from ._losses import LOSSES
from ._utils import get_study_summary
from .main import run_hpo

__all__ = ["run_hpo", "LOSSES", "get_study_summary"]


