"""Pre-built architectures for reservoir computing models.

This module provides complete model architectures for different types of
Echo State Networks and other reservoir computing approaches.
"""

from typing import Any, Dict, Union

import tensorflow as tf

# Import only what's needed to avoid circular imports
from keras_reservoir_computing.layers import (
    SelectiveExponentiation,
)
from keras_reservoir_computing.layers.builders import (
    ESNReservoir_builder,
    ReadOut_builder,
)
from keras_reservoir_computing.layers.config_layers import load_user_config

# Default configurations
ESN_RESERVOIR_CONFIG: Dict[str, Any] = {
    "input_dim": 0,
    "leak_rate": 1.0,
    "activation": "tanh",
    "input_initializer": {"name": "zeros", "params": {}},
    "feedback_initializer": {
        "name": "RandomInputInitializer",
        "params": {"input_scaling": None, "seed": None},
    },
    "feedback_bias_initializer": {"name": "zeros", "params": {}},
    "kernel_initializer": {
        "name": "RandomRecurrentInitializer",
        "params": {
            "density": 0.01,
            "spectral_radius": 0.9,
            "seed": None,
        },
    },
}


READOUT_CONFIG: Dict[str, Any] = {
    "kind": "ridge",
    "units": 1,
    "alpha": 0.1,
    "trainable": False,
}


def classic_ESN(
    units: int,
    reservoir_config: Union[str, Dict[str, Any]] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, Dict[str, Any]] = READOUT_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "classic_ESN",
    dtype: str = "float32",
) -> tf.keras.Model:
    """
    Build a classic Echo State Network (ESN) model.

    Parameters
    ----------
    units : int
        Number of units in the reservoir.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, dict], optional
        Configuration for the readout layer. Can be a path to a JSON file or a dictionary.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.
    name : str, optional
        Name for the model. Default is "classic_ESN".
    dtype : str, optional
        Data type for the model. Default is "float32".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the classic ESN.

    Notes
    -----
    - The architecture is: Input Layer -> ESN Reservoir -> Readout Layer
    """
    # Load config from file if string is provided
    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    if isinstance(readout_config, str):
        readout_config = load_user_config(readout_config)

    # Create the input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)

    # Override units parameter in reservoir config
    reservoir_overrides = {"units": units, "feedback_dim": features, "dtype": dtype}
    reservoir = ESNReservoir_builder(user_config=reservoir_config, overrides=reservoir_overrides)(input_layer)

    # Concatenate input with reservoir
    concatenation = tf.keras.layers.Concatenate(dtype=dtype)([input_layer, reservoir])

    readout_overrides = {"dtype": dtype}
    readout = ReadOut_builder(readout_config, overrides=readout_overrides)(concatenation)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=readout, name=name, dtype=dtype)
    return model


def Ott_ESN(
    units: int,
    reservoir_config: Union[str, Dict[str, Any]] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, Dict[str, Any]] = READOUT_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "Ott_ESN",
    dtype: str = "float32",
) -> tf.keras.Model:
    """
    Build Ott's ESN model with state augmentation.

    This model follows the architecture proposed by Edward Ott, which augments
    reservoir states by squaring even-indexed units and concatenating with input.

    Parameters
    ----------
    units : int
        Number of units in the reservoir.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, dict], optional
        Configuration for the readout layer. Can be a path to a JSON file or a dictionary.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.
    name : str, optional
        Name for the model. Default is "Ott_ESN".
    dtype : str, optional
        Data type for the model. Default is "float32".
    Returns
    -------
    tf.keras.Model
        A Keras Model representing Ott's ESN.

    Notes
    -----
    - The architecture augments the reservoir output with squared values of even-indexed units
    - The augmented reservoir output is concatenated with the input before the readout layer

    References
    ----------
    .. E. Ott, J. Pathak, B. Hunt, M. Girvan, and Z. Lu, "Model-Free Prediction of Large
       Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach,"
       Phys. Rev. Lett., vol. 120, no. 2, p. 024102, Jan. 2018.
    """
    # Load config from file if string is provided
    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    if isinstance(readout_config, str):
        readout_config = load_user_config(readout_config)

    # Create input layer
    feedback_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)

    # Build reservoir with overridden parameters
    reservoir_overrides = {"units": units, "feedback_dim": features, "dtype": dtype}
    reservoir = ESNReservoir_builder(user_config=reservoir_config, overrides=reservoir_overrides)(
        feedback_layer
    )

    # Augment reservoir output by squaring even-indexed units
    selective_exponentiation = SelectiveExponentiation(
        index=0,
        exponent=2.0,
        dtype=dtype,
    )(reservoir)

    # Concatenate original input with augmented reservoir output
    concat = tf.keras.layers.Concatenate(dtype=dtype)([feedback_layer, selective_exponentiation])

    # Create readout layer
    readout_overrides = {"dtype": dtype}
    readout = ReadOut_builder(readout_config, overrides=readout_overrides)(concat)

    # Build and return model
    model = tf.keras.Model(inputs=feedback_layer, outputs=readout, name=name, dtype=dtype)
    return model


def headless_ESN(
    units: int,
    reservoir_config: Union[str, Dict[str, Any]] = ESN_RESERVOIR_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "headless_ESN",
    dtype: str = "float32",
) -> tf.keras.Model:
    """
    Build an ESN model with no readout layer.

    This model can be used to study the dynamics of the reservoir by applying different transformations to the reservoir states.
    """
    # Load config from file if string is provided
    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)

    # Build reservoir
    reservoir_overrides = {"units": units, "feedback_dim": features, "dtype": dtype}
    reservoir = ESNReservoir_builder(user_config=reservoir_config, overrides=reservoir_overrides)(input_layer)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=reservoir, name=name, dtype=dtype)
    return model
