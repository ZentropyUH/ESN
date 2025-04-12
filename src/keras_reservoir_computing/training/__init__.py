"""Training utilities for reservoir computing models.

This module provides tools for training reservoir computing models.
"""

from .training import ReservoirTrainer

__all__ = ["ReservoirTrainer"]


def __dir__() -> list[str]:
    return __all__