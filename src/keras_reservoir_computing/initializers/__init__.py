"""Initializers for reservoir computing.

This module provides specialized initializers for input layers and reservoirs in ESN models.
"""
from . import input_initializers, recurrent_initializers

from .input_initializers import (
    ChebyshevInitializer,
    ChessboardInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
)
from .recurrent_initializers import (
    BarabasiAlbertGraphInitializer,
    CompleteGraphInitializer,
    ConnectedRandomMatrixInitializer,
    DigitalChaosInitializer,
    ErdosRenyiGraphInitializer,
    KleinbergSmallWorldGraphInitializer,
    MultiCliqueGraphInitializer,
    NewmanWattsStrogatzGraphInitializer,
    RegularGraphInitializer,
    RandomRecurrentInitializer,
    TernaryInitializer,
    WattsStrogatzGraphInitializer,
)

__all__ = ["input_initializers", "recurrent_initializers"]

__all__ += [
    "ChebyshevInitializer",
    "ChessboardInitializer",
    "PseudoDiagonalInitializer",
    "RandomBinaryInitializer",
    "RandomInputInitializer",
]

__all__ += [
    "BarabasiAlbertGraphInitializer",
    "CompleteGraphInitializer",
    "ConnectedRandomMatrixInitializer",
    "DigitalChaosInitializer",
    "ErdosRenyiGraphInitializer",
    "KleinbergSmallWorldGraphInitializer",
    "MultiCliqueGraphInitializer",
    "NewmanWattsStrogatzGraphInitializer",
    "RegularGraphInitializer",
    "RandomRecurrentInitializer",
    "TernaryInitializer",
    "WattsStrogatzGraphInitializer",
]

def __dir__() -> list[str]:
    return __all__