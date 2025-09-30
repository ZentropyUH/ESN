# =============================================================
# krc/hpo/_objective.py
# =============================================================
"""Objective factory - converts user callbacks into an Optuna objective."""
import gc
import logging
from typing import Any, Callable, List, Mapping, MutableMapping

import optuna
import tensorflow as tf
from keras import backend as K

from keras_reservoir_computing.forecasting import warmup_forecast
from keras_reservoir_computing.layers.readouts.base import ReadOut
from keras_reservoir_computing.training import (
    ReservoirTrainer,  # local import to avoid circular deps
)

from ._losses import LossProtocol

__all__ = ["build_objective"]

logger = logging.getLogger(__name__)


def _validate_data_dict(data: Mapping[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys are present in the data dictionary."""
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required keys in data dictionary: {', '.join(missing_keys)}. "
            f"Required keys: {', '.join(required_keys)}"
        )


def build_objective(
    *,
    model_creator: Callable[..., tf.keras.Model],
    search_space: Callable[[optuna.trial.Trial], MutableMapping[str, Any]],
    loss_fn: LossProtocol,
    data_loader: Callable[[], Mapping[str, Any]],
    penalty_value: float = 1e10,
) -> Callable[[optuna.trial.Trial], float]:
    """Return an Optuna objective function (closure)."""

    # --------------------------------------------------------------
    # Objective closure
    # --------------------------------------------------------------
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Build hyper-parameter dict & model, then run the trainer.
        """
        # ----------------------------------------------------------
        # Build hyper-parameter dict & model
        # ----------------------------------------------------------
        try:
            params = search_space(trial)
        except optuna.TrialPruned:
            raise  # bubble up
        except Exception as exc:  # pragma: no cover - user error
            logger.exception("Search space callable failed.")
            raise optuna.TrialPruned() from exc

        try:
            model = model_creator(**params)
        except Exception:
            return penalty_value

        # ----------------------------------------------------------
        # Load data once per trial - user provides the splits
        # ----------------------------------------------------------
        data = data_loader(trial)

        return _run_custom_trainer(model, data, loss_fn)

    return objective


# ------------------------------------------------------------------
# Helpers - trainer variants
# ------------------------------------------------------------------

def _run_custom_trainer(
    model: tf.keras.Model,
    data: Mapping[str, Any],
    loss_fn: LossProtocol,
) -> float:
    """Run ReservoirTrainer and evaluate loss_fn on held-out data."""
    # Validate required data keys
    _validate_data_dict(
        data,
        ["transient", "train", "train_target", "ftransient", "val", "val_target"]
    )

    # Extract required data
    transient_data = data["transient"]
    train_data = data["train"]
    train_target = data["train_target"]
    ftransient = data["ftransient"]
    val_data = data["val"]
    val_target = data["val_target"]

    # Extract readout targets mapping if provided, otherwise create a default one
    if "readout_targets" in data:
        readout_targets = data["readout_targets"]
    else:
        # Find all ReadOut layers in the model
        readout_layers = [layer for layer in model.layers if isinstance(layer, ReadOut)]

        if len(readout_layers) == 0:
            raise ValueError("No ReadOut layers found in the model")
        elif len(readout_layers) == 1:
            # If only one readout layer, use train_target for it
            readout_targets = {readout_layers[0].name: train_target}
        else:
            raise ValueError(
                "Multiple ReadOut layers found in the model. "
                "Please provide a 'readout_targets' mapping in the data dictionary."
            )

    # Topological readout fitting
    trainer = ReservoirTrainer(
        model=model, 
        readout_targets=readout_targets
        )
    trainer.fit_readout_layers(
        warmup_data=transient_data, 
        input_data=train_data
        )

    # Forecast, we only care about the predictions not the states
    preds, _ = warmup_forecast(
        model=model,
        warmup_data=ftransient,
        forecast_data=val_data,
        horizon=val_target.shape[1],
        show_progress=False
    )

    # Align length - just in case
    T = min(preds.shape[1], val_target.shape[1])
    return float(loss_fn(val_target[:, :T, :], preds[:, :T, :]))



# --------------------------------------------------------------
# Cleanup helpers - minimise GPU memory leaks between trials
# --------------------------------------------------------------

def _cleanup() -> None:  # noqa: D401
    # Perform a more careful cleanup to preserve model weights
    gc.collect()
    K.clear_session()


# Ensure cleanup runs after each trial regardless of success
def _wrap(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            _cleanup()
    return inner
globals()[_run_custom_trainer.__name__] = _wrap(_run_custom_trainer)  # type: ignore[misc]


