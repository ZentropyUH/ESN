"""Define some general utility functions."""

from contextlib import contextmanager
from time import time
from typing import Optional, Union, Generator
import numpy as np


def create_rng(
    seed: Optional[Union[int, np.random.Generator]] = None
) -> np.random.Generator:
    """
    Create and return a NumPy random number generator (RNG).

    Parameters
    ----------
    seed : int, np.random.Generator, or None, optional
        Seed or an existing RNG. If an integer, a new generator is created from that seed.
        If it's already an instance of ``np.random.Generator``, it is returned as is.
        If None, a new generator is created without a fixed seed (non-deterministic).

    Returns
    -------
    np.random.Generator
        A NumPy generator instance.

    Raises
    ------
    ValueError
        If ``seed`` is neither an integer, a ``np.random.Generator``, nor None.

    Notes
    -----
    - This function consolidates the creation of a random generator from an integer seed,
      an existing generator, or from no seed at all.
    - Using a fixed seed makes your random operations reproducible.
    """
    if isinstance(seed, int):
        rg = np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        rg = seed
    elif seed is None:
        rg = np.random.default_rng()
    else:
        raise ValueError("Seed must be an integer, a np.random.Generator, or None.")

    return rg


def lyap_ks(i_th: int, L_period: float) -> float:
    """
    Estimate the i-th largest Lyapunov Time for the Kuramoto-Sivashinsky (KS) PDE model.

    This approximation is from:
    "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arXiv:1902.09651v1"

    Parameters
    ----------
    i_th : int
        The index (1-based) of the desired Lyapunov Time (i-th largest).
    L_period : float
        The spatial period of the KS system.

    Returns
    -------
    float
        The estimated value of the i-th Lyapunov Time.

    Notes
    -----
    - The expression is an empirically derived approximation, so exact accuracy
      may vary.
    - The formula used is:
        .. math::
           0.093 - 0.94 \\times \\frac{(i_{th} - 0.39)}{L_{period}}.
    """
    return 0.093 - 0.94 * ((i_th - 0.39) / L_period)


@contextmanager
def timer(task_name: str = "Task", log: bool = True) -> Generator[None, None, None]:
    """
    Measure and optionally log the execution time of a code block.

    Parameters
    ----------
    task_name : str, optional
        A descriptive name for the task being measured. Default is "Task".
    log : bool, optional
        If True, prints timing information to stdout.

    Yields
    ------
    None
        This context yields no value; it simply measures and logs time.

    Examples
    --------
    >>> with timer("Data Loading", log=True):
    ...     data = load_big_file("some_file.csv")
    ...
    # Prints something like: "Data Loading took: 0.45 seconds."
    """
    if log:
        print(f"\n{task_name}...\n")
    start = time()
    yield
    end = time()
    if log:
        duration = round(end - start, 2)
        print(f"{task_name} took: {duration} seconds.\n")


__all__ = [
    "timer",
    "lyap_ks",
    "create_rng",
]


def __dir__():
    return __all__
