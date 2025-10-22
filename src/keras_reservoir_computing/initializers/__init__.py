"""
Initializers for reservoir computing layers.

This module consolidates all available initializer classes for use in the
library's Keras-based reservoir layers. It provides two distinct categories
of initializers:

Input initializers
------------------
Define the connection weights from the input/feedback layers to the reservoir, controlling how external/feedback signals enter the network.

Recurrent initializers
----------------------
Define the internal recurrent connection weights of the reservoir, determining the network's internal dynamics.


Available input initializers
----------------------------
- :class:`BinaryBalancedInitializer`
- :class:`ChebyshevInitializer`
- :class:`ChessboardInitializer`
- :class:`DendrocycleInputInitializer`
- :class:`PseudoDiagonalInitializer`
- :class:`RandomBinaryInitializer`
- :class:`RandomInputInitializer`

Available recurrent initializers
--------------------------------
- :class:`BarabasiAlbertGraphInitializer`
- :class:`CompleteGraphInitializer`
- :class:`ConnectedRandomMatrixInitializer`
- :class:`DendrocycleGraphInitializer`
- :class:`DigitalChaosInitializer`
- :class:`ErdosRenyiGraphInitializer`
- :class:`KleinbergSmallWorldGraphInitializer`
- :class:`MultiCycleGraphInitializer`
- :class:`NewmanWattsStrogatzGraphInitializer`
- :class:`RegularGraphInitializer`
- :class:`SpectralCascadeGraphInitializer`
- :class:`RandomRecurrentInitializer`
- :class:`TernaryInitializer`
- :class:`WattsStrogatzGraphInitializer`
"""
from . import input_initializers, recurrent_initializers
from .input_initializers import (
    BinaryBalancedInitializer,
    ChebyshevInitializer,
    ChessboardInitializer,
    DendrocycleInputInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
)
from .recurrent_initializers import (
    BarabasiAlbertGraphInitializer,
    CompleteGraphInitializer,
    ConnectedRandomMatrixInitializer,
    DendrocycleGraphInitializer,
    DigitalChaosInitializer,
    ErdosRenyiGraphInitializer,
    KleinbergSmallWorldGraphInitializer,
    MultiCycleGraphInitializer,
    NewmanWattsStrogatzGraphInitializer,
    RandomRecurrentInitializer,
    RegularGraphInitializer,
    SpectralCascadeGraphInitializer,
    TernaryInitializer,
    WattsStrogatzGraphInitializer,
)

__all__ = ["input_initializers", "recurrent_initializers"]

__all__ += [
    "BinaryBalancedInitializer",
    "ChebyshevInitializer",
    "ChessboardInitializer",
    "DendrocycleInputInitializer",
    "PseudoDiagonalInitializer",
    "RandomBinaryInitializer",
    "RandomInputInitializer",
]

__all__ += [
    "BarabasiAlbertGraphInitializer",
    "CompleteGraphInitializer",
    "ConnectedRandomMatrixInitializer",
    "DendrocycleGraphInitializer",
    "DigitalChaosInitializer",
    "ErdosRenyiGraphInitializer",
    "KleinbergSmallWorldGraphInitializer",
    "MultiCycleGraphInitializer",
    "NewmanWattsStrogatzGraphInitializer",
    "RegularGraphInitializer",
    "SpectralCascadeGraphInitializer",
    "RandomRecurrentInitializer",
    "TernaryInitializer",
    "WattsStrogatzGraphInitializer",
]

def __dir__() -> list[str]:
    return __all__