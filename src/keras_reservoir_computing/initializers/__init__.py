"""Initializers for reservoir computing.

This module provides specialized initializers for input layers and reservoirs in ESN models.
"""
from . import input_initializers, recurrent_initializers

from .input_initializers import (
    ChebyshevInitializer,
    DenseBinaryInitializer,
    PseudoDiagonalInitializer,
)
from .recurrent_initializers import (
    BarabasiAlbertGraphInitializer,
    CompleteGraphInitializer,
    ConnectedRandomMatrixInitializer,
    DigitalChaosInitializer,
    ErdosRenyiGraphInitializer,
    KleinbergSmallWorldGraphInitializer,
    NewmanWattsStrogatzGraphInitializer,
    RegularGraphInitializer,
    TernaryInitializer,
    WattsStrogatzGraphInitializer,
)

__all__ = ["input_initializers", "recurrent_initializers"]

__all__ += ["ChebyshevInitializer", "DenseBinaryInitializer", "PseudoDiagonalInitializer"]

__all__ += [
    "BarabasiAlbertGraphInitializer",
    "CompleteGraphInitializer",
    "ConnectedRandomMatrixInitializer",
    "DigitalChaosInitializer",
    "ErdosRenyiGraphInitializer",
    "KleinbergSmallWorldGraphInitializer",
    "NewmanWattsStrogatzGraphInitializer",
    "RegularGraphInitializer",
    "TernaryInitializer",
    "WattsStrogatzGraphInitializer",
]

def __dir__() -> list[str]:
    return __all__