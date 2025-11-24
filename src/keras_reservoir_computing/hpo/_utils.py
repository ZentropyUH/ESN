# =============================================================
# krc/hpo/_utils.py
# =============================================================
"""Utility functions for hyperparameter optimization.

This module provides helper functions for managing Optuna studies,
cleaning up memory, and creating study names. It includes functions
for summarizing study results, monitoring loss functions, and
ensuring clean TF sessions.
"""
from __future__ import annotations

import functools
import gc
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import optuna

# Silence TF logs before any potential Keras/TensorFlow usage.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if TYPE_CHECKING:
    import tensorflow as tf  # for type hints only

__all__ = ["make_study_name", "get_study_summary", "monitor_name", "_with_cleanup"]


def make_study_name(model_creator: Callable[..., "tf.keras.Model"]) -> str:
    src = inspect.getsourcefile(model_creator) or "<interactive>"
    func = getattr(model_creator, "__name__", "model_creator")
    return f"{Path(src).stem}:{func}"


def get_study_summary(study: optuna.Study, top_n: int = 5) -> str:
    """Generate a summary of the Optuna study.

    Parameters
    ----------
    study : optuna.Study
        The Optuna study to summarize.
    top_n : int, default=5
        Number of top trials to include in the summary.

    Returns
    -------
    str
        A formatted string containing the study summary.
    """
    lines = []
    lines.append("Study Summary")
    lines.append("=" * 50)
    lines.append(f"Study Name: {study.study_name}")

    total_trials = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    lines.append(f"Total Trials: {total_trials}")
    lines.append(f"Completed: {completed}")
    lines.append(f"Pruned: {pruned}")
    lines.append(f"Failed: {failed}")
    lines.append("")

    if completed > 0:
        lines.append("Best Trial:")
        lines.append("-" * 50)
        lines.append(f"Trial Number: {study.best_trial.number}")
        lines.append(f"Value: {study.best_value:.6f}")
        lines.append("Parameters:")
        for k, v in study.best_params.items():
            lines.append(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        lines.append(f"Top {min(top_n, len(sorted_trials))} Trials:")
        lines.append("-" * 50)
        for i, trial in enumerate(sorted_trials[:top_n], 1):
            lines.append(f"{i}. Trial {trial.number}: {trial.value:.6f}")

    return "\n".join(lines)


def monitor_name(spec, base):
    """Get the name of a monitor loss function.

    Parameters
    ----------
    spec : str | Callable
        The monitor specification.
    base : Callable
        The base monitor function.

    Returns
    -------
    str
        The name of the monitor loss function.
    """
    if isinstance(spec, str):
        return spec
    if isinstance(base, functools.partial):
        fn = base.func
        return getattr(fn, "__name__", "custom_loss")
    return getattr(base, "__name__", "custom_loss")


# ------------------------------------------------------------------
# Memory management
# ------------------------------------------------------------------


def _cleanup() -> None:
    """Garbage collection and clear Keras session without importing TF at module import time."""
    gc.collect()
    # Import here to avoid TF import during module load.
    from keras import backend as K

    K.clear_session()


def _with_cleanup(fn: Callable) -> Callable:
    """Wrap a function with memory cleanup.

    Parameters
    ----------
    fn : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            _cleanup()

    return wrapper
