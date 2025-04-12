"""Pre-built model architectures for reservoir computing.

This module provides ready-to-use model architectures for common reservoir computing tasks.
"""
# Import from architectures directly to fix the lookup
from .architectures import (
    Ott_ESN,
    classic_ESN,
    ensemble_with_mean_ESN,
    residual_stacked_ESN,
)

__all__ = ["Ott_ESN", "classic_ESN", "ensemble_with_mean_ESN", "residual_stacked_ESN"]
