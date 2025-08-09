"""Layers for reservoir computing.

Overview
--------
Compact collection of Keras-compatible layers used to build Echo State
Networks (ESNs) and related reservoir-computing models. The package provides:

- Reservoirs: fixed-weight recurrent layers that produce rich state sequences
  from inputs (ESN).
- Readouts: closed-form regression heads that map reservoir states to targets.
- Utility layers: small building blocks to manipulate features and ensembles.

Layers
------
FeaturePartitioner
    Split the feature axis into ``partitions`` overlapping slices with optional
    circular wrapping. Useful for structured sub-reservoirs or multi-branch
    processing.
OutliersFilteredMean
    Remove outlier samples (per batch and timestep) using Z-score or IQR and
    return the mean over the remaining samples.
SelectiveDropout
    Zero a fixed subset of features everywhere according to a boolean mask.
SelectiveExponentiation
    Raise either even or odd feature indices (chosen by ``index`` parity) to a
    given ``exponent``; leave the rest unchanged.

Readouts
--------
MoorePenroseReadout
    Pseudoinverse-based linear readout with optional ridge-like regularization
    ``alpha``; solved in closed form (internal float64 for stability).
RidgeReadout
    L2-regularized linear readout solved with a conjugate-gradient routine;
    optionally trainable after fitting.

Reservoirs
----------
ESNCell
    Echo State Network cell with leak, fixed weights and configurable
    initializers for feedback, input and recurrent kernels.
ESNReservoir
    ``tf.keras.layers.RNN`` wrapper around ``ESNCell``; stateful, returns full
    sequences, supports feedback-only or feedback+input modes.

Examples
--------
Minimal ESN with a linear readout::

    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers import ESNReservoir, RidgeReadout
    >>> x = tf.random.normal((8, 50, 1))   # (batch, time, feedback_dim)
    >>> reservoir = ESNReservoir(units=100, feedback_dim=1)
    >>> states = reservoir(x)               # (8, 50, 100)
    >>> readout = RidgeReadout(units=1, alpha=1.0)
    >>> readout.build(states.shape)  # or: readout.fit(states, y)

Notes
-----
- Reservoir layers are non-trainable by default and always return sequences.
- Readouts operate in float64 internally for numerical stability.
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
