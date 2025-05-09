"""Specialized layers for reservoir computing.

This module contains custom layers used in reservoir computing models,
including reservoirs, readout layers, and various utility layers.
"""

from .custom_layers import (
    FeaturePartitioner,
    OutliersFilteredMean,
    SelectiveDropout,
    SelectiveExponentiation,
)
from .readouts import (
    MoorePenroseReadout,
    RidgeReadout,
)
from .reservoirs import ESNCell, ESNReservoir

__all__ = [
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "SelectiveDropout",
    "SelectiveExponentiation",
]

__all__ += [
    "ESNCell",
    "ESNReservoir",
]

__all__ += [
    "MoorePenroseReadout",
    "RidgeReadout",
]


def __dir__() -> list[str]:
    return __all__
