# =============================================================
# krc/hpo/_objective.py
# =============================================================
"""Objective factory - converts user callbacks into an Optuna objective.

This module provides the core machinery for transforming user-defined callbacks
(model creator, search space, data loader) into a fully-functional Optuna
objective that can be optimized. It handles model creation, training, forecasting,
and evaluation with robust error handling and memory management.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, MutableMapping, Optional

# Silence TF/absl logs in parent *and* child before any TF import happens.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cloudpickle as cp
import numpy as np
import optuna
from optuna import TrialPruned

from ._losses import LossProtocol
from .validators import _validate_tensor_shapes
from .worker import _child_worker

if TYPE_CHECKING:
    import tensorflow as tf

__all__ = ["build_objective"]
logger = logging.getLogger(__name__)

# Default timeout for child process execution (in seconds)
# Can be overridden via environment variable KRC_HPO_PROCESS_TIMEOUT
DEFAULT_PROCESS_TIMEOUT = int(os.environ.get("KRC_HPO_PROCESS_TIMEOUT", "7200"))  # 2 hours default


# ------------------------------------------------------------------
# Objective builder
# ------------------------------------------------------------------

def build_objective(
    *,
    model_creator: Callable[..., "tf.keras.Model"],
    search_space: Callable[[optuna.trial.Trial], MutableMapping[str, Any]],
    loss_fn: LossProtocol,
    data_loader: Callable[[optuna.trial.Trial], Mapping[str, Any]],
    validate_shapes: bool = True,
    monitor_specs: Optional[List[tuple[str, LossProtocol]]] = None,
    clip_value: Optional[float] = None,
    prune_on_clip: bool = False,
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
    monitor_specs : Optional[List[tuple[str, LossProtocol]]], default=None
        Monitor loss functions to evaluate during optimization. If None, no monitoring is done.
    clip_value : Optional[float], default=None
        Value to clip the monitor losses to. If None, no clipping is done.
    prune_on_clip : bool, default=False
        Whether to prune trials where any monitor loss is clipped. Otherwise, the trial is penalized by the clipped value.

    Returns
    -------
    Callable[[optuna.trial.Trial], float]
        The objective function for Optuna to optimize.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        # 1) Hyperparameters
        params = search_space(trial)

        # 2) Data
        data = data_loader(trial)
        if validate_shapes:
            _validate_tensor_shapes(data)

        # 3) Run in a fresh process (TF isolated)
        value, attrs, status, msg = _run_in_fresh_process(
            model_creator=model_creator,
            params=params,
            data=data,
            loss_fn=loss_fn,
            monitor_specs=monitor_specs,
            clip_value=clip_value,
            prune_on_clip=prune_on_clip,
        )
        if attrs:
            for k, v in attrs.items():
                trial.set_user_attr(k, v)

        if status == "ok":
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite loss encountered: {value}")
            return float(value)
        if status == "pruned":
            raise TrialPruned(msg or "Pruned in child process")

        logger.error(f"Trial {trial.number}: child error: {msg}")
        raise RuntimeError(msg or "Training or evaluation failed in child")

    return objective


# ------------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------------


def _run_in_fresh_process(
    *,
    model_creator: Callable[..., "tf.keras.Model"],
    params: Mapping[str, Any],
    data: Mapping[str, Any],
    loss_fn: LossProtocol,
    monitor_specs: Optional[List[tuple[str, LossProtocol]]],
    clip_value: Optional[float],
    prune_on_clip: bool,
    timeout: Optional[float] = None,
) -> tuple[float, dict | None, str, Optional[str]]:
    """
    Run model creation + training + evaluation in a clean Python process.

    Parameters
    ----------
    model_creator : Callable[..., tf.keras.Model]
        Function that creates a fresh model given hyperparameters.
    params : Mapping[str, Any]
        Dictionary containing the hyperparameters.
    data : Mapping[str, Any]
        Dictionary containing all required data splits.
    loss_fn : LossProtocol
        Loss function to evaluate predictions.
    monitor_specs : Optional[List[tuple[str, LossProtocol]]], default=None
        Monitor loss functions to evaluate during optimization. If None, no monitoring is done.
    clip_value : Optional[float], default=None
        Value to clip the monitor losses to. If None, no clipping is done.
    prune_on_clip : bool, default=False
        Whether to prune trials where any monitor loss is clipped. Otherwise, the trial is penalized by the clipped value.
    timeout : Optional[float], default=None
        Timeout in seconds for the child process. If None, uses DEFAULT_PROCESS_TIMEOUT.
        If the process exceeds this timeout, it will be terminated.

    Returns
    -------
    (value, attrs, status, message)
    - value : float
    - attrs : dict | None   (user attrs collected in child)
    - status: "ok" | "pruned" | "error"
    - message: Optional[str]
    """
    if timeout is None:
        timeout = DEFAULT_PROCESS_TIMEOUT

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    # Always serialize with cloudpickle (no dotted-path fallback).
    mc_blob: bytes = cp.dumps(model_creator)

    p = ctx.Process(
        target=_child_worker,
        args=(q, mc_blob, params, data, loss_fn, monitor_specs, clip_value, prune_on_clip),
        daemon=False,
    )
    p.start()

    # Use join with timeout to prevent infinite hangs
    p.join(timeout=timeout)

    # Check if process is still alive (timed out)
    if p.is_alive():
        # Process timed out, but check queue first in case it finished but cleanup is hanging
        if not q.empty():
            try:
                status, value, attrs, msg = q.get_nowait()
                # We got a result, but process is still alive - might be stuck in cleanup
                # Terminate it anyway for cleanup
                logger.warning("Child process timed out but result available, terminating process")
                try:
                    p.terminate()
                    p.join(timeout=5.0)
                    if p.is_alive():
                        p.kill()
                        p.join()
                except Exception:
                    pass  # Ignore termination errors
                return (value, attrs, status, msg)
            except queue.Empty:
                pass  # Queue is empty, proceed with timeout handling

        # No result in queue, process truly timed out
        logger.error(f"Child process exceeded timeout of {timeout}s, terminating")
        try:
            p.terminate()
            # Give it a moment to terminate gracefully
            p.join(timeout=5.0)
            if p.is_alive():
                # Force kill if still alive
                p.kill()
                p.join()
        except Exception as exc:
            logger.error(f"Error terminating child process: {exc}")
        return (float("inf"), None, "error", f"Child process exceeded timeout of {timeout}s")

    # Process finished (not alive), check exit code
    if p.exitcode != 0 and p.exitcode is not None:
        logger.warning(f"Child process exited with code {p.exitcode}")

    # Get result from queue
    if q.empty():
        return (float("inf"), None, "error", f"Child exited with code {p.exitcode} (no result in queue)")

    try:
        # Get result with a short timeout to avoid blocking forever
        status, value, attrs, msg = q.get(timeout=5.0)
        return (value, attrs, status, msg)
    except queue.Empty:
        logger.error("Queue is empty or timeout waiting for result")
        return (float("inf"), None, "error", "Timeout waiting for result from child process")
    except Exception as exc:
        logger.error(f"Error retrieving result from queue: {exc}")
        return (float("inf"), None, "error", f"Failed to get result from queue: {exc}")


