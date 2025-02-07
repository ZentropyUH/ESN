"""Define some general utility functions."""

from contextlib import contextmanager
from time import time
import numpy as np


def create_rng(seed: int = None) -> np.random.Generator:
    """
    Creates a random number generator from a seed.

    Parameters
    ----------
    seed : int, np.random.Generator, or None
        Seed for the random number generator. If None, a new generator is created.

    Returns
    -------
    np.random.Generator
        Random

    Notes
    -----
    - If seed is an integer, a new generator is created using the seed.
    - If seed is a generator, it is returned as is.
    - If seed is None, a new generator is created without a seed.
    """
    # create random generator from seed
    if isinstance(seed, int):
        rg = np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        rg = seed
    elif seed is None:
        rg = np.random.default_rng()
    else:
        raise ValueError("Seed must be an integer, a random generator, or None.")

    return rg


def lyap_ks(i_th, L_period):
    """Estimation of the i-th largest Lyapunov Time of the KS model.

    Args:
        i_th (int): The i-th largest Lyapunov Time.

        L_period (int): The period of the system.

    Returns:
        float: The estimated i-th largest Lyapunov Time.

    Taken from the paper:
        "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arxiv:1902.09651v1"
    """
    # This approximation is taken from the above paper. Verify veracity.
    return 0.093 - 0.94 * (i_th - 0.39) / L_period


@contextmanager
def timer(task_name: str = "Task", log: bool = True):
    """
    Context manager to measure the time of a task.

    Args:
        task_name (str): Name of the task to measure.
        log (bool): Whether to log the time taken.

    Returns:
        None

    Example:
        >>> with self.timer("Some Task"):
        >>>     # Code to measure
        Will print the time taken to execute the code block.
    """
    if log:
        print(f"\n{task_name}...\n")
    start = time()
    yield
    end = time()
    if log:
        print(f"{task_name} took: {round(end - start, 2)} seconds.\n")


__all__ = [
    # utils
    "timer",
    "lyap_ks",
    "create_rng",
]


def __dir__():
    return __all__
