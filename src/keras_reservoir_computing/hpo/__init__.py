# =============================================================
# krc/hpo/__init__.py
# =============================================================
"""Hyper-parameter optimisation (HPO) for reservoir-computing models.

This subpackage provides a compact, high-level wrapper around `Optuna` to
search hyper-parameters for Keras-based reservoir models. It exposes a single
entry point for running studies and a small registry of multi-step forecasting
losses tailored for chaotic and non-chaotic dynamical systems.

Public API
----------
- :func:`run_hpo` — fully managed Optuna study runner.
- :data:`LOSSES` — registry mapping short keys to built-in multi-step losses.

Overview
--------
``run_hpo`` constructs an :class:`optuna.Study`, builds an objective with
your callbacks, and executes the optimisation. You provide three callables:

1. ``model_creator``: builds a fresh :class:`tf.keras.Model` for each trial
  and accepts the parameters returned by ``search_space`` as keyword args.

2. ``search_space(trial)``: registers parameters via ``trial.suggest_*``
  and returns the mapping passed to ``model_creator``.

3. ``data_loader(trial)``: returns the arrays required by the trainer.

Trainer
-------------
Topological readout fitting using :class:`ReservoirTrainer`,
  then a generative forecast with :func:`warmup_forecast`.
  The ``data_loader`` must return the keys:

    - ``"transient"``

    - ``"train"``

    - ``"train_target"``

    - ``"ftransient"``

    - ``"val"``

    - ``"val_target"``

  Optionally, provide ``"readout_targets"`` mapping when multiple :class:`ReadOut` layers are present.


Losses
------
The :data:`LOSSES` registry contains objectives operating on batched, multi-
step forecasts with shapes ``(B, T, D)``. Pick by string key or pass a
callable. Built-ins:

- ``"horizon"`` → :func:`forecast_horizon_loss`:
  Negative log of the valid forecast horizon w.r.t. a threshold.

- ``"lyap"`` → :func:`lyapunov_weighted_loss`: Geometric-mean error
  weighted by ``exp(-lle * t * dt)``; robust in chaotic regimes.

- ``"multi_step"`` → :func:`multi_step_loss`: Sum of geometric-mean
  error across time.

- ``"standard"`` → :func:`standard_loss`: Mean of geometric-mean
  error across time.

- ``"efh"`` → :func:`expected_forecast_horizon`: Smooth, differentiable
  proxy for the forecast horizon
  (more negative ⇒ longer expected horizon). Accepts ``threshold`` and ``softness``. Used by default and recommended.

Selecting a loss
----------------
Choose either:

- A key from :data:`LOSSES`.

- A custom callable following ``LossProtocol`` (NumPy path).

  Signature:

  >>> def loss(y_true: np.ndarray, y_pred: np.ndarray, /, *args, **kwargs) -> float:

  Requirements:
  - ``y_true`` and ``y_pred`` shaped ``(B, T, D)``.

  - Return a Python ``float`` to minimise.

  - Should be pure/deterministic and vectorised over batch/time.

  - Extra keyword args go via ``loss_params`` in :func:`run_hpo`.

Study naming
------------
If ``study_name`` is not provided, a default is generated via
:func:`make_study_name` as ``<file>:<func>``.

Example
-------
>>> import tensorflow as tf
>>> from keras_reservoir_computing.hpo import run_hpo
>>> def model_creator(units: int, leak_rate: float) -> tf.keras.Model:
...     # return a fresh, uncompiled model
...     ...
>>> def search_space(trial):
...     return {
...         "units": trial.suggest_int("units", 128, 1024, step=128),
...         "leak_rate": trial.suggest_float("leak_rate", 0.05, 0.5),
...     }
>>> def data_loader(trial):
...     return {
...         "transient": X_trans, "train": X_train, "train_target": y_train,
...         "ftransient": X_warm, "val": X_val, "val_target": y_val,
...     }
>>> study = run_hpo(
...     model_creator, search_space,
...     n_trials=50, data_loader=data_loader,
...     loss="lyap", loss_params={"lle": 1.2, "dt": 1.0},
... )

Notes
-----
- Return a fresh (uncompiled) model per trial.
- The objective returns a scalar ``float``; lower is better.
- When model construction fails, the objective returns ``penalty_value`` (default
  ``1e10``), allowing constraints to be expressed in the search space.
- The default sampler is :class:`optuna.samplers.TPESampler` with
  ``multivariate=True``.

See Also
--------
:mod:`keras_reservoir_computing.hpo._objective`,
:mod:`keras_reservoir_computing.hpo._losses`,
:class:`ReservoirTrainer`,
:func:`warmup_forecast`.
"""
from ._losses import LOSSES
from .main import run_hpo

__all__ = ["run_hpo", "LOSSES"]


