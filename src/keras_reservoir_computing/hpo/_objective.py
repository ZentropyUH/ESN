# =============================================================
# krc/hpo/_objective.py
# =============================================================
"""Objective factory - converts user callbacks into an Optuna objective.

This module provides the core machinery for transforming user-defined callbacks
(model creator, search space, data loader) into a fully-functional Optuna
objective that can be optimized. It handles model creation, training, forecasting,
and evaluation with robust error handling and memory management.
"""
import gc
import logging
import traceback
from typing import Any, Callable, List, Mapping, MutableMapping, Optional

import numpy as np
import optuna
import tensorflow as tf
from keras import backend as K

from optuna import TrialPruned

from keras_reservoir_computing.forecasting import warmup_forecast
from keras_reservoir_computing.layers.readouts.base import ReadOut
from keras_reservoir_computing.training import ReservoirTrainer

from ._losses import LossProtocol

__all__ = ["build_objective"]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Validation utilities
# ------------------------------------------------------------------

def _validate_data_dict(data: Mapping[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys are present in the data dictionary.

    Parameters
    ----------
    data : Mapping[str, Any]
        The data dictionary to validate.
    required_keys : List[str]
        List of required keys that must be present.

    Raises
    ------
    KeyError
        If any required keys are missing.
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required keys in data dictionary: {', '.join(missing_keys)}. "
            f"Required keys: {', '.join(required_keys)}"
        )


def _validate_tensor_shapes(
    data: Mapping[str, Any],
    trial: Optional[optuna.trial.Trial] = None
) -> None:
    """Validate that data tensors have consistent shapes.

    Parameters
    ----------
    data : Mapping[str, Any]
        The data dictionary containing tensors.
    trial : Optional[optuna.trial.Trial]
        The current trial (for logging purposes).

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent.
    """
    # Check batch dimensions are consistent
    batch_keys = ["transient", "train", "train_target", "ftransient", "val", "val_target"]
    batch_sizes = {}

    for key in batch_keys:
        if key in data:
            tensor = data[key]
            if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
                batch_sizes[key] = tensor.shape[0]

    if len(set(batch_sizes.values())) > 1:
        logger.warning(
            f"Inconsistent batch sizes detected: {batch_sizes}. "
            "This may cause issues during training."
        )


# ------------------------------------------------------------------
# Objective builder
# ------------------------------------------------------------------

def build_objective(
    *,
    model_creator: Callable[..., tf.keras.Model],
    search_space: Callable[[optuna.trial.Trial], MutableMapping[str, Any]],
    loss_fn: LossProtocol,
    data_loader: Callable[[optuna.trial.Trial], Mapping[str, Any]],
    validate_shapes: bool = True,
) -> Callable[[optuna.trial.Trial], float]:
    """Build an Optuna objective function from user-defined callbacks.

    This function creates a closure that wraps the model creation, training,
    and evaluation process into a single objective function that Optuna can
    optimize. It provides robust error handling, logging, and memory management.

    Parameters
    ----------
    model_creator : Callable[..., tf.keras.Model]
        Function that creates a fresh model given hyperparameters.
    search_space : Callable[[optuna.trial.Trial], MutableMapping[str, Any]]
        Function that defines the hyperparameter search space.
    loss_fn : LossProtocol
        Loss function to evaluate model performance.
    data_loader : Callable[[optuna.trial.Trial], Mapping[str, Any]]
        Function that loads and returns the training/validation data.
    validate_shapes : bool, optional
        Whether to validate tensor shapes (default: True).

    Returns
    -------
    Callable[[optuna.trial.Trial], float]
        The objective function for Optuna to optimize.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        """Objective function for a single trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The current Optuna trial.

        Returns
        -------
        float
            The loss value (lower is better).
        """
        # ----------------------------------------------------------
        # 1. Generate hyperparameters
        # ----------------------------------------------------------
        try:
            params = search_space(trial)
            logger.debug(f"Trial {trial.number}: Generated parameters {params}")
        except TrialPruned:
            logger.debug(f"Trial {trial.number}: Pruned during search space generation")
            raise  # Bubble up pruned trials
        except Exception as exc:
            logger.error(
                f"Trial {trial.number}: Search space generation failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
            raise TrialPruned(f"Search space generation failed: {exc}")

        # ----------------------------------------------------------
        # 2. Create model
        # ----------------------------------------------------------
        try:
            model = model_creator(**params)
            logger.debug(f"Trial {trial.number}: Model created successfully")
        except Exception as exc:
            logger.warning(
                f"Trial {trial.number}: Model creation failed with parameters {params}: {exc}"
            )
            raise TrialPruned(f"Model creation failed: {exc}")

        # ----------------------------------------------------------
        # 3. Load data
        # ----------------------------------------------------------
        try:
            data = data_loader(trial)
            logger.debug(f"Trial {trial.number}: Data loaded successfully")

            # Validate shapes if requested
            if validate_shapes:
                _validate_tensor_shapes(data, trial)

        except Exception as exc:
            logger.error(
                f"Trial {trial.number}: Data loading failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
            raise TrialPruned(f"Data loading failed: {exc}")

        # ----------------------------------------------------------
        # 4. Train and evaluate
        # ----------------------------------------------------------
        try:
            loss_value = _train_and_evaluate(model, data, loss_fn, trial)

            # Validate loss value
            if not np.isfinite(loss_value):
                logger.warning(
                    f"Trial {trial.number}: Non-finite loss value {loss_value}, "
                    f"returning penalty"
                )
                raise TrialPruned(f"Non-finite loss encountered: {loss_value}")

            logger.debug(f"Trial {trial.number}: Loss = {loss_value:.6f}")
            return loss_value

        except Exception as exc:
            logger.error(
                f"Trial {trial.number}: Training/evaluation failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
            raise TrialPruned(f"Training or evaluation failed: {exc}")

    return objective


# ------------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------------

def _train_and_evaluate(
    model: tf.keras.Model,
    data: Mapping[str, Any],
    loss_fn: LossProtocol,
    trial: Optional[optuna.trial.Trial] = None,
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
    # ----------------------------------------------------------
    # 1. Validate required data keys
    # ----------------------------------------------------------
    required_keys = ["transient", "train", "train_target", "ftransient", "val"]
    _validate_data_dict(data, required_keys)

    # ----------------------------------------------------------
    # 2. Extract data
    # ----------------------------------------------------------
    transient_data = data["transient"]
    train_data = data["train"]
    train_target = data["train_target"]
    ftransient = data["ftransient"]
    val_data = data["val"]
    external_inputs = data["external_inputs"] if "external_inputs" in data else ()

    # ----------------------------------------------------------
    # 3. Determine readout targets
    # ----------------------------------------------------------
    if "readout_targets" in data:
        readout_targets = data["readout_targets"]
    else:
        readout_targets = _infer_readout_targets(model, train_target)

    # ----------------------------------------------------------
    # 4. Train readout layers
    # ----------------------------------------------------------
    trainer = ReservoirTrainer(
        model=model,
        readout_targets=readout_targets,
        log=False,  # Keep quiet during HPO
        )

    trainer.fit_readout_layers(
        warmup_data=transient_data,
        input_data=train_data,
        )

    # ----------------------------------------------------------
    # 5. Generate forecasts
    # ----------------------------------------------------------
    preds, _ = warmup_forecast(
        model=model,
        warmup_data=ftransient,
        horizon=val_data.shape[1],
        external_inputs=external_inputs,
        show_progress=False,
        states=False,  # Don't track states for efficiency
    )

    # ----------------------------------------------------------
    # 6. Evaluate loss
    # ----------------------------------------------------------
    # Align lengths in case of minor shape mismatches
    T = min(preds.shape[1], val_data.shape[1])

    # Convert to numpy if needed
    if hasattr(preds, 'numpy'):
        preds_np = preds[:, :T, :].numpy()
    else:
        preds_np = preds[:, :T, :]

    if hasattr(val_data, 'numpy'):
        val_data_np = val_data[:, :T, :].numpy()
    else:
        val_data_np = val_data[:, :T, :]

    loss_value = loss_fn(val_data_np, preds_np)

    return float(loss_value)


def _infer_readout_targets(
    model: tf.keras.Model,
    train_target: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    """Infer readout targets from model structure.

    Automatically detects ReadOut layers in the model and assigns
    training targets to them. For single ReadOut models, uses the
    provided train_target. For multiple ReadOuts, raises an error
    requesting explicit specification.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing ReadOut layers.
    train_target : tf.Tensor
        The training target tensor.

    Returns
    -------
    Mapping[str, tf.Tensor]
        Dictionary mapping ReadOut layer names to target tensors.

    Raises
    ------
    ValueError
        If no ReadOut layers are found or if multiple ReadOuts exist
        without explicit target specification.
    """
    readout_layers = [
        layer for layer in model.layers
        if isinstance(layer, ReadOut)
    ]

    if len(readout_layers) == 0:
        raise ValueError(
            "No ReadOut layers found in the model. "
            "Ensure your model contains at least one ReadOut layer."
        )
    elif len(readout_layers) == 1:
        # Single readout - use the provided target
        return {readout_layers[0].name: train_target}
    else:
        # Multiple readouts - require explicit specification
        raise ValueError(
            f"Multiple ReadOut layers found: {[layer.name for layer in readout_layers]}. "
            "Please provide a 'readout_targets' mapping in the data dictionary "
            "that maps each ReadOut layer name to its corresponding target."
        )


# ------------------------------------------------------------------
# Memory management
# ------------------------------------------------------------------

def _cleanup() -> None:
    """Clean up memory after trial completion.

    Performs garbage collection and clears Keras session to minimize
    memory leaks between trials. This is especially important for
    GPU memory management.
    """
    gc.collect()
    K.clear_session()


def _with_cleanup(fn: Callable) -> Callable:
    """Decorator that ensures cleanup runs after function execution.

    Parameters
    ----------
    fn : Callable
        The function to wrap with cleanup.

    Returns
    -------
    Callable
        Wrapped function that cleans up after execution.
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            _cleanup()
    return wrapper


# Wrap the training function with cleanup
_train_and_evaluate = _with_cleanup(_train_and_evaluate)


