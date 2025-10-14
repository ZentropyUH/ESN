"""Forecasting utilities for reservoir computing models.

This module provides functions for generating forecasts with trained reservoir models.
"""

from .forecasting import (
    warmup_forecast,
    window_forecast,
)

__all__ = ["warmup_forecast", "window_forecast"]


def __dir__() -> list[str]:
    return __all__