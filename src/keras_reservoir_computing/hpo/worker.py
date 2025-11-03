"""Worker process for the Optuna objective.


This module provides the core machinery for running the model creation,
training, and evaluation in a separate process. It handles memory management,
error handling, and communication with the main process.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# (Optional, but recommended to avoid CPU oversubscription per child)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, List, Mapping, Optional

import cloudpickle as cp
import numpy as np
from optuna import Trial, TrialPruned

from keras_reservoir_computing.forecasting import warmup_forecast
from keras_reservoir_computing.training import ReservoirTrainer

from ._losses import LossProtocol
from ._utils import _with_cleanup
from .validators import _infer_readout_targets, _validate_data_dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tensorflow as tf


class _AttrSink:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}
    def set_user_attr(self, k, v) -> None:
        self.attrs[k] = v
    def report(self, *_, **__) -> None:
        pass
    def should_prune(self) -> bool:
        return False

def _log_monitors(
    trial: Optional[Trial],
    monitors: Optional[List[tuple[str, LossProtocol]]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Log monitor losses to the trial.

    Parameters
    ----------
    trial : Optional[optuna.trial.Trial]
        The current trial (for logging purposes).
    monitors : Optional[List[tuple[str, LossProtocol]]], default=None
        Monitor loss functions to evaluate during optimization. If None, no monitoring is done.
    y_true : np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted values.
    """
    if trial is None or not monitors:
        return
    for name, fn in monitors:
        try:
            v = float(fn(y_true, y_pred))
            trial.set_user_attr(name, v)
        except Exception as exc:
            logger.debug(f"Monitor '{name}' failed: {exc}")



@_with_cleanup
def _train_and_evaluate(
    model: "tf.keras.Model",
    data: Mapping[str, Any],
    loss_fn: LossProtocol,
    trial: Optional[Trial] = None,
    monitor_specs: Optional[List[tuple[str, LossProtocol]]] = None,
    clip_value: Optional[float] = None,
    prune_on_clip: bool = False,
) -> float:
    """Train the model and evaluate it on validation data.

    This function performs the complete training and evaluation pipeline:
    1. Validates data dictionary
    2. Extracts readout targets
    3. Trains readout layers using ReservoirTrainer
    4. Generates forecasts using warmup_forecast
    5. Evaluates using the loss function

    Parameters
    ----------
    model : tf.keras.Model
        The model to train and evaluate.
    data : Mapping[str, Any]
        Dictionary containing all required data splits.
    loss_fn : LossProtocol
        Loss function to evaluate predictions.
    trial : Optional[optuna.trial.Trial]
        The current trial (for logging purposes).
    monitor_specs : Optional[List[tuple[str, LossProtocol]]], default=None
        Monitor loss functions to evaluate during optimization. If None, no monitoring is done.
    clip_value : Optional[float], default=None
        Value to clip the monitor losses to. If None, no clipping is done.
    prune_on_clip : bool, default=False
        Whether to prune trials where any monitor loss is clipped. Otherwise, the trial is penalized by the clipped value.

    Returns
    -------
    float
        The computed loss value.

    Raises
    ------
    KeyError
        If required data keys are missing.
    ValueError
        If model structure is invalid or data shapes are incompatible.
    """
    # 1) Data keys
    required_keys = ["transient", "train", "train_target", "ftransient", "val"]
    _validate_data_dict(data, required_keys)

    # 2) Extract
    transient_data = data["transient"]
    train_data = data["train"]
    train_target = data["train_target"]
    ftransient = data["ftransient"]
    val_data = data["val"]
    external_inputs = data.get("external_inputs", None)

    # 3) Targets per ReadOut
    if "readout_targets" in data:
        readout_targets = data["readout_targets"]
    else:
        readout_targets = _infer_readout_targets(model, train_target)

    # 4) Train readout layers
    trainer = ReservoirTrainer(model=model, readout_targets=readout_targets, log=False)
    trainer.fit_readout_layers(warmup_data=transient_data, input_data=train_data)

    # 5) Forecast
    preds, _ = warmup_forecast(
        model=model,
        warmup_data=ftransient,
        horizon=val_data.shape[1],
        external_inputs=external_inputs,
        show_progress=False,
        states=False,
    )

    T = min(preds.shape[1], val_data.shape[1])
    preds_np = preds[:, :T, :].numpy() if hasattr(preds, "numpy") else preds[:, :T, :]
    val_np = val_data[:, :T, :].numpy() if hasattr(val_data, "numpy") else val_data[:, :T, :]

    # 6) Loss + clip
    raw = float(loss_fn(val_np, preds_np))
    clipped = (clip_value is not None) and (raw > 0) and (raw > clip_value)
    final = clip_value if clipped else raw

    # Diagnostics
    if trial is not None:
        trial.set_user_attr("loss/raw", raw)
        trial.set_user_attr("loss/clipped", final)
        trial.set_user_attr("loss/was_clipped", clipped)

    _log_monitors(trial, monitor_specs, val_np, preds_np)

    if prune_on_clip and clipped:
        raise TrialPruned(f"Pruned trial due to clip (raw={raw:.6g} > {clip_value})")

    return float(final)


def _child_worker(
    q_: mp.Queue,
    mc_blob: bytes,
    params: Mapping[str, Any],
    data: Mapping[str, Any],
    loss_fn: LossProtocol,
    monitor_specs: Optional[List[tuple[str, LossProtocol]]],
    clip_value: Optional[float],
    prune_on_clip: bool,
) -> None:
    import tensorflow as tf  # imported only in the child
    from keras import backend as K

    try:
        # GPU memory growth (best-effort)
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception as exc:
                logger.debug(f"Could not set memory growth for device {g}: {exc}")

        # Reconstruct model_creator
        model_creator = cp.loads(mc_blob)

        # Collect attrs without Optuna in the child
        sink = _AttrSink()
        model = model_creator(**params)

        val = _train_and_evaluate(
            model=model,
            data=data,
            loss_fn=loss_fn,
            trial=sink,
            monitor_specs=monitor_specs,
            clip_value=clip_value,
            prune_on_clip=prune_on_clip,
        )
        q_.put(("ok", float(val), sink.attrs, None))
    except TrialPruned as tp:
        q_.put(("pruned", float("inf"), None, str(tp)))
    except Exception as exc:
        q_.put(("error", float("inf"), None, repr(exc)))
    finally:
        try:
            K.clear_session()
        except Exception as exc:
            logger.debug(f"K.clear_session() failed during cleanup: {exc}")


