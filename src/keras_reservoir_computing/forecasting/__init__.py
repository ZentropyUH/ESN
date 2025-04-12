"""Forecasting utilities for reservoir computing models.

This module provides functions for generating forecasts with trained reservoir models.
"""

from .forecasting import (
    forecast,
    warmup_forecast,
)

__all__ = ["forecast", "warmup_forecast"]


def __dir__() -> list[str]:
    return __all__