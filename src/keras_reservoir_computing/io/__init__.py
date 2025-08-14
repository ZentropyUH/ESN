"""
This module provides utilities for loading and saving keras objects (layers, initializers, optimizers, etc.), configurations and default configurations.

Available functions
-------------------
- :func:`load_config` loads a configuration from a YAML/JSON file or a dictionary.
- :func:`load_default_config` loads a default configuration from the ``io/defaults`` directory, provided the name of the config.
- :func:`load_object` loads a Keras object (layer, initializer, optimizer, etc.) from a file or a dictionary.

Default configurations
----------------------

Configurations are located in the ``io/defaults`` directory, saved as YAML/JSON files.

- ``reservoir``
- ``readout``
"""
from .loaders import load_object, load_config, load_default_config

__all__ = ["load_object", "load_config", "load_default_config"]

def __dir__() -> list[str]:
    return __all__