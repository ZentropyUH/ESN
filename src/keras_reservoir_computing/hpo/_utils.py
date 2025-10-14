# =============================================================
# krc/hpo/_utils.py
# =============================================================
"""Utility functions for hyperparameter optimization.

This module provides helper functions for the HPO module, including
study naming conventions and result analysis utilities.
"""
import inspect
import logging
from pathlib import Path
from typing import Callable

import optuna
import tensorflow as tf

__all__ = ["make_study_name", "get_study_summary"]

logger = logging.getLogger(__name__)


def make_study_name(
    model_creator: Callable[..., tf.keras.Model],
) -> str:
    """Generate a default study name for Optuna hyperparameter optimization.

    The study name is constructed as: ``<file>:<func>`` where:
    
    - ``<file>`` is the stem of the source file containing the model creator
    - ``<func>`` is the name of the model creator function

    This naming convention helps identify and organize studies when multiple
    experiments are run in parallel or stored in the same database.

    Parameters
    ----------
    model_creator : Callable[..., tf.keras.Model]
        Function that creates the model to be optimized.

    Returns
    -------
    str
        The generated study name in the format ``<file>:<func>``.

    Examples
    --------
    >>> def my_model(units, leak_rate):
    ...     # defined in reservoir_models.py
    ...     return model
    >>> make_study_name(my_model)
    'reservoir_models:my_model'
    """
    src = inspect.getsourcefile(model_creator) or "<interactive>"
    func = model_creator.__name__
    return f"{Path(src).stem}:{func}"


def get_study_summary(
    study: optuna.Study,
    top_n: int = 5,
) -> str:
    """Generate a human-readable summary of an Optuna study.

    Creates a formatted summary including:
    
    - Total number of trials
    - Best trial information (value and parameters)
    - Top N trials ranked by value
    - Study statistics

    Parameters
    ----------
    study : optuna.Study
        The completed or in-progress Optuna study to summarize.
    top_n : int, default=5
        Number of top trials to include in the summary.

    Returns
    -------
    str
        Formatted summary string.

    Examples
    --------
    >>> study = run_hpo(model_creator, search_space, n_trials=100, data_loader=loader)
    >>> print(get_study_summary(study))
    Study Summary
    =============
    Study Name: reservoir_models:my_model
    Total Trials: 100
    Completed: 98
    Pruned: 2
    Failed: 0
    
    Best Trial:
    -----------
    Value: 42.35
    Parameters:
      units: 800
      spectral_radius: 1.23
      leak_rate: 0.15
    ...
    """
    lines = []
    lines.append("Study Summary")
    lines.append("=" * 50)
    lines.append(f"Study Name: {study.study_name}")
    
    # Count trials by state
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
        # Best trial
        lines.append("Best Trial:")
        lines.append("-" * 50)
        lines.append(f"Trial Number: {study.best_trial.number}")
        lines.append(f"Value: {study.best_value:.6f}")
        lines.append("Parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Top N trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        
        lines.append(f"Top {min(top_n, len(sorted_trials))} Trials:")
        lines.append("-" * 50)
        for i, trial in enumerate(sorted_trials[:top_n], 1):
            lines.append(f"{i}. Trial {trial.number}: {trial.value:.6f}")
            
    return "\n".join(lines)
