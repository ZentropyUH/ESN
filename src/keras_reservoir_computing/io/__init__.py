"""Utilities for loading and saving Keras objects and configurations.

This module provides utilities for loading and saving Keras objects (layers,
initializers, optimizers, etc.), configurations and default configurations.

Available functions
-------------------
- :func:`load_config` : Load a configuration from a YAML/JSON file or a dictionary.
- :func:`load_default_config` : Load a default configuration from the ``io/defaults`` directory.
- :func:`load_object` : Load a Keras object (layer, initializer, optimizer, etc.) from a file or a dictionary.
- :func:`load_reservoir_config` : Load and validate a reservoir configuration.
- :func:`load_readout_config` : Load and validate a readout configuration.

Configuration models
--------------------
- :class:`LayerConfig` : Base configuration model for Keras layers.
- :class:`ReservoirConfig` : Configuration model for reservoir layers.
- :class:`ReadoutConfig` : Configuration model for readout layers.

Default configurations
----------------------
Configurations are located in the ``io/defaults`` directory, saved as YAML/JSON files.

- ``reservoir``
- ``readout``
"""

from .config_models import (
    LayerConfig,
    ReadoutConfig,
    ReservoirConfig,
)
from .loaders import (
    load_config,
    load_default_config,
    load_object,
    load_readout_config,
    load_reservoir_config,
)

__all__ = [
    "load_object",
    "load_config",
    "load_default_config",
    "load_reservoir_config",
    "load_readout_config",
    "LayerConfig",
    "ReservoirConfig",
    "ReadoutConfig",
]


def __dir__() -> list[str]:
    return __all__
