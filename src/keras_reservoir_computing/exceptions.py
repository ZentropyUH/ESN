"""Custom exceptions for keras_reservoir_computing.

This module provides a hierarchy of custom exceptions for better error handling
and debugging throughout the library.
"""


class ReservoirComputingError(Exception):
    """Base exception for all reservoir computing errors.

    All custom exceptions in this library inherit from this class,
    making it easy to catch any library-specific errors.
    """

    pass


class ConfigurationError(ReservoirComputingError):
    """Raised when configuration is invalid or inconsistent.

    Examples
    --------
    - Invalid parameter combinations
    - Missing required configuration keys
    - Type mismatches in configuration values
    """

    pass


class FittingError(ReservoirComputingError):
    """Raised when readout layer fitting fails.

    Examples
    --------
    - Numerical instability in ridge regression
    - Singular matrix in conjugate gradient solver
    - Data shape mismatches during fitting
    """

    pass


class ShapeMismatchError(ReservoirComputingError):
    """Raised when tensor shapes don't match expected dimensions.

    Examples
    --------
    - Input tensor has wrong number of features
    - Batch size inconsistencies
    - Sequence length mismatches
    """

    pass


class InitializationError(ReservoirComputingError):
    """Raised when initializer configuration or execution fails.

    Examples
    --------
    - Invalid spectral radius
    - Incompatible initializer parameters
    - Graph generation failures
    """

    pass


__all__ = [
    "ReservoirComputingError",
    "ConfigurationError",
    "FittingError",
    "ShapeMismatchError",
    "InitializationError",
]
