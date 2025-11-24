"""Pre-built model architectures for reservoir computing.

This module provides ready-to-use model architectures for common reservoir computing tasks.
"""

# Import from architectures directly to fix the lookup
from .architectures import (
    Ott_ESN,
    classic_ESN,
    headless_ESN,
    linear_ESN,
    ensemble_model,
)

__all__ = ["Ott_ESN", "classic_ESN", "headless_ESN", "linear_ESN", "ensemble_model"]
