"""Keras Reservoir Computing (KRC) library.

A TensorFlow/Keras implementation of reservoir computing with a focus on Echo State Networks (ESNs).
This library provides a flexible framework for creating, training, and evaluating reservoir computing models
integrated with the Keras API.
"""
from . import (
    analysis,
    callbacks,
    forecasting,
    hpo,
    initializers,
    layers,
    models,
    training,
    utils,
)

__all__ = [
    "analysis",
    "callbacks",
    "forecasting",
    "hpo",
    "initializers",
    "layers",
    "models",
    "training",
    "utils",
]

def __dir__() -> list[str]:
    return __all__
