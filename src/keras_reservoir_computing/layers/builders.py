from typing import Dict, Optional, Union

import tensorflow as tf

from keras_reservoir_computing.layers import ESNReservoir
from keras_reservoir_computing.layers.config import (
    get_default_params,
    load_user_config,
    merge_with_defaults,
)
from keras_reservoir_computing.layers.readouts.base import ReadOut
from keras_reservoir_computing.layers.readouts.moorepenrose import MoorePenroseReadout
from keras_reservoir_computing.layers.readouts.ridge import RidgeSVDReadout


def ESNReservoir_builder(
    user_config: Union[str, dict], overrides: Optional[Dict] = None
) -> ESNReservoir:
    """
    Constructs an Echo State Network (ESN) reservoir layer from a user-defined configuration.

    Parameters
    ----------
    user_config : Union[str, dict]
        Either a path to a JSON/YAML configuration file or a dictionary specifying the
        parameters for the ESN reservoir. The expected configuration dictionary format includes:

        Example:
        --------
        {
            "units": 10,
            "feedback_dim": 1,
            "input_dim": 0,
            "leak_rate": 1.0,
            "activation": "tanh",
            "input_initializer": {"name": "glorot_uniform", "params": {}},
            "feedback_initializer": {"name": "PseudoDiagonalInitializer", "params": {"sigma": 0.5, "binarize": False, "seed": None}},
            "feedback_bias_initializer": {"name": "glorot_uniform", "params": {}},
            "kernel_initializer": {"name": "WattsStrogatzGraphInitializer", "params": {"k": 6, "p": 0.2, "directed": True, "self_loops": True, "tries": 100, "spectral_radius": 0.9, "seed": None}},
            "dtype": "float32"
        }

    Returns
    -------
    ESNReservoir
        An initialized ESN reservoir layer with the specified configuration.

    Notes
    -----
    - If a path to a configuration file is provided, the function automatically loads it.
    - The final configuration merges the user-defined parameters with default values.
    - The reservoir supports multiple initializers for different weight matrices, which can be:
      - Keras built-in initializers (e.g., "glorot_uniform")
      - Custom initializers prefixed with "krc>" (e.g., "krc>PseudoDiagonalInitializer")
    - The function ensures that all initializers specified in the configuration are correctly resolved.
    - The spectral radius is adjusted on the effective weight matrix [(1-lr) I + lr W] if the leak rate is below 1.

    Raises
    ------
    ValueError
        If an initializer is not found in either Keras built-ins or custom-registered initializers.
    """

    user_config = load_user_config(config=user_config)
    default_config = get_default_params(cls=ESNReservoir)
    final_config = merge_with_defaults(
        default_params=default_config,
        user_params=user_config,
        override_params=overrides,
    )

    # Adjust the spectral radius of the effective weight matrix.
    if "params" in final_config["kernel_initializer"]:
        if "spectral_radius" in final_config["kernel_initializer"]["params"]:
            effective_sr = final_config["kernel_initializer"]["params"][
                "spectral_radius"
            ]
            if effective_sr is not None:  # The default values is usually None
                leak_rate = final_config.get("leak_rate", 1.0)
                if effective_sr < 1 - leak_rate:
                    raise ValueError(
                        "The spectral radius must be greater than 1 - leak_rate."
                    )
                W_sr = (effective_sr - 1 + leak_rate) / leak_rate
                final_config["kernel_initializer"]["params"]["spectral_radius"] = W_sr

    # Convert each initializer from {name, params} to a Keras-recognized initializer
    for key in [
        "input_initializer",
        "feedback_initializer",
        "feedback_bias_initializer",
        "kernel_initializer",
    ]:
        if key in final_config and isinstance(final_config[key], dict):
            # E.g. {"name": "PseudoDiagonalInitializer", "params": {...}}
            name = final_config[key]["name"]
            params = final_config[key].get("params", {})
            try:
                # First attempt: standard retrieval
                final_config[key] = tf.keras.initializers.get(
                    {"class_name": name, "config": params}
                )
            except ValueError:
                try:
                    # Second attempt: Try with package prefix if it's a custom-registered initializer
                    final_config[key] = tf.keras.initializers.get(
                        {"class_name": f"krc>{name}", "config": params}
                    )
                except ValueError:
                    raise ValueError(
                        f"Initializer '{name}' not found, either as keras built-in nor as custom class."
                    )

    # Instantiate the layer
    # units = final_config.pop("units", 100)
    layer = ESNReservoir(**final_config)
    return layer


def ReadOut_builder(
    user_config: Union[Dict, str], overrides: Optional[Dict] = None
) -> ReadOut:
    """
    Constructs a readout layer for an Echo State Network (ESN) based on the specified type.

    Parameters
    ----------
    user_config : Union[Dict, str]
        Either a path to a JSON/YAML configuration file or a dictionary specifying the
        parameters for the readout layer.

        Example:
        --------
        {
            "kind": "ridge",
            "units": 100,
            "alpha": 0.1,
            "trainable": False
        }

    Returns
    -------
    ReadOut
        An instantiated readout layer of the specified type.

    Notes
    -----
    - If a path to a configuration file is provided, the function automatically loads it.
    - The function merges user-defined parameters with default values specific to the chosen readout type.
    - The readout layer can be of two types: `ridge` or `mpenrose`. The default is `ridge`.
    - Each readout method has distinct configurable parameters:
      - `ridge` uses Ridge regression with SVD, allowing for regularization (`alpha`).
      - `mpenrose` computes the Moore-Penrose pseudoinverse, useful for direct least-squares solutions.

    Raises
    ------
    ValueError
        If the specified `kind` is not recognized.
    """

    user_config = load_user_config(config=user_config)

    kind = user_config.pop("kind", "ridge")

    if kind == "ridge":
        readout_class = RidgeSVDReadout
    elif kind == "mpenrose":
        readout_class = MoorePenroseReadout
    else:
        raise ValueError(f"Readout kind '{kind}' not recognized.")
    default_config = get_default_params(cls=readout_class)
    final_config = merge_with_defaults(
        default_params=default_config,
        user_params=user_config,
        override_params=overrides,
    )

    layer = readout_class(**final_config)
    return layer


__all__ = ["ESNReservoir_builder", "ReadOut_builder"]


def __dir__() -> list:
    return __all__
