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

Graph initializers
------------------
Define the adjacency matrices of the reservoir, controlling the network's topology.

Available input initializers
----------------------------
- :class:`BinaryBalancedInitializer`
- :class:`ChebyshevInitializer`
- :class:`ChessboardInitializer`
- :class:`DendrocycleInputInitializer`
- :class:`OppositeAnchorsInputInitializer`
- :class:`PseudoDiagonalInitializer`
- :class:`RandomBinaryInitializer`
- :class:`RandomInputInitializer`
- :class:`RingWindowInputInitializer`


Available recurrent initializers
--------------------------------
- :class:`BarabasiAlbertGraphInitializer`
- :class:`ChordDendrocycleGraphInitializer`
- :class:`CompleteGraphInitializer`
- :class:`ConnectedRandomMatrixInitializer`
- :class:`DendrocycleGraphInitializer`
- :class:`DigitalChaosInitializer`
- :class:`ErdosRenyiGraphInitializer`
- :class:`KleinbergSmallWorldGraphInitializer`
- :class:`MultiCycleGraphInitializer`
- :class:`NewmanWattsStrogatzGraphInitializer`
- :class:`RegularGraphInitializer`
- :class:`RingChordGraphInitializer`
- :class:`SimpleCycleJumpsGraphInitializer`
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
    OppositeAnchorsInputInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
    RingWindowInputInitializer,
)
from .recurrent_initializers import (
    BarabasiAlbertGraphInitializer,
    ChordDendrocycleGraphInitializer,
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
    RingChordGraphInitializer,
    SimpleCycleJumpsGraphInitializer,
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
    "OppositeAnchorsInputInitializer",
    "PseudoDiagonalInitializer",
    "RandomBinaryInitializer",
    "RandomInputInitializer",
    "RingWindowInputInitializer",
]

__all__ += [
    "BarabasiAlbertGraphInitializer",
    "ChordDendrocycleGraphInitializer",
    "CompleteGraphInitializer",
    "ConnectedRandomMatrixInitializer",
    "DendrocycleGraphInitializer",
    "DigitalChaosInitializer",
    "ErdosRenyiGraphInitializer",
    "KleinbergSmallWorldGraphInitializer",
    "MultiCycleGraphInitializer",
    "NewmanWattsStrogatzGraphInitializer",
    "RegularGraphInitializer",
    "RingChordGraphInitializer",
    "SimpleCycleJumpsGraphInitializer",
    "SpectralCascadeGraphInitializer",
    "RandomRecurrentInitializer",
    "TernaryInitializer",
    "WattsStrogatzGraphInitializer",
]

def __dir__() -> list[str]:
    return __all__