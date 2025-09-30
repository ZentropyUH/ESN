# =============================================================
# krc/hpo/_utils.py
# =============================================================
"""Miscellaneous utils - kept minimal to avoid bloat."""
import inspect
from pathlib import Path
from typing import Callable

import tensorflow as tf


def make_study_name(
    model_creator: Callable[..., tf.keras.Model],
) -> str:
    """
    Generate a default study name for Optuna hyperparameter optimization.

    The study name is constructed as: ``<file>:<func>`` where:
    - ``<file>`` is the stem of the source file containing the model creator
    - ``<func>`` is the name of the model creator function

    Parameters
    ----------
    model_creator : Callable[..., tf.keras.Model]
        Function that creates the model to be optimized

    Returns
    -------
    str
        The generated study name
    """
    src = inspect.getsourcefile(model_creator) or "<interactive>"
    func = model_creator.__name__
    return f"{Path(src).stem}:{func}"
