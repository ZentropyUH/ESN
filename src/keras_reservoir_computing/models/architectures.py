"""Pre-built architectures for reservoir computing models.

This module provides complete model architectures for different types of 
Echo State Networks and other reservoir computing approaches.
"""

from typing import Any, Dict, List, Union

import tensorflow as tf

# Import only what's needed to avoid circular imports
from keras_reservoir_computing.layers import (
    OutliersFilteredMean,
    SelectiveExponentiation,
)
from keras_reservoir_computing.layers.builders import (
    ESNReservoir_builder,
    ReadOut_builder,
)
from keras_reservoir_computing.layers.config import load_user_config

# Default configurations
ESN_RESERVOIR_CONFIG: Dict[str, Any] = {
    "units": 10,
    "feedback_dim": 1,
    "input_dim": 0,
    "leak_rate": 1.0,
    "activation": "tanh",
    "input_initializer": {"name": "glorot_uniform", "params": {}},
    "feedback_initializer": {
        "name": "PseudoDiagonalInitializer",
        "params": {"sigma": 0.5, "binarize": False, "seed": None},
    },
    "feedback_bias_initializer": {"name": "glorot_uniform", "params": {}},
    "kernel_initializer": {
        "name": "WattsStrogatzGraphInitializer",
        "params": {
            "k": 6,
            "p": 0.2,
            "directed": True,
            "self_loops": True,
            "tries": 100,
            "spectral_radius": 0.9,
            "seed": None,
        },
    },
}


READOUT_CONFIG: Dict[str, Any] = {
    "kind": "ridge",
    "units": 100,
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
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    # Override units parameter in reservoir config
    overrides = {"units": units, "feedback_dim": features}
    reservoir = ESNReservoir_builder(reservoir_config, overrides=overrides)(input_layer)

    # Create readout layer
    readout_config_copy = (
        readout_config.copy() if isinstance(readout_config, dict) else readout_config
    )
    if isinstance(readout_config_copy, dict):
        readout_config_copy.update(
            {"units": features}
        )  # Ensure output dimensionality matches input
    readout = ReadOut_builder(readout_config_copy)(reservoir)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=readout, name=name)
    return model


def Ott_ESN(
    units: int,
    reservoir_config: Union[str, Dict[str, Any]] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, Dict[str, Any]] = READOUT_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "Ott_ESN",
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
    feedback_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    # Build reservoir with overridden parameters
    overrides = {"units": units, "feedback_dim": features}
    reservoir = ESNReservoir_builder(reservoir_config, overrides=overrides)(
        feedback_layer
    )

    # Augment reservoir output by squaring even-indexed units
    selective_exponentiation = SelectiveExponentiation(
        index=0,
        exponent=2.0,
    )(reservoir)

    # Concatenate original input with augmented reservoir output
    concat = tf.keras.layers.Concatenate()([feedback_layer, selective_exponentiation])

    # Create readout layer
    readout_config_copy = (
        readout_config.copy() if isinstance(readout_config, dict) else readout_config
    )
    if isinstance(readout_config_copy, dict):
        readout_config_copy.update(
            {"units": features}
        )  # Ensure output dimensionality matches input
    readout = ReadOut_builder(readout_config_copy)(concat)

    # Build and return model
    model = tf.keras.Model(inputs=feedback_layer, outputs=readout, name=name)
    return model


def ensemble_with_mean_ESN(
    units: int,
    ensemble_size: int,
    reservoir_config: Union[str, Dict[str, Any]] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, Dict[str, Any]] = READOUT_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "ensemble_with_mean_ESN",
) -> tf.keras.Model:
    """
    Build an ensemble of ESNs with outlier-filtered mean aggregation.

    This model creates multiple independent ESN models and combines their
    outputs using a mean that filters outliers.

    Parameters
    ----------
    units : int
        Number of units in each reservoir.
    ensemble_size : int
        Number of reservoir models in the ensemble.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, dict], optional
        Configuration for the readout layers. Can be a path to a JSON file or a dictionary.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.
    name : str, optional
        Name for the model. Default is "ensemble_with_mean_ESN".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the ensemble ESN.

    Notes
    -----
    - Each ensemble member follows Ott's ESN architecture
    - The final output is an outlier-filtered mean of all ensemble members
    """
    # Load config from file if string is provided
    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    if isinstance(readout_config, str):
        readout_config = load_user_config(readout_config)

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    # Set up reservoir configuration
    reservoir_config_copy = (
        reservoir_config.copy()
        if isinstance(reservoir_config, dict)
        else reservoir_config
    )
    if isinstance(reservoir_config_copy, dict):
        reservoir_config_copy["units"] = units
        reservoir_config_copy["feedback_dim"] = features

    # Create ensemble members
    reservoirs = [
        ESNReservoir_builder(reservoir_config_copy)(input_layer)
        for _ in range(ensemble_size)
    ]

    # Apply state augmentation to each reservoir
    exponentiations = [
        SelectiveExponentiation(index=0, exponent=2.0)(reservoir)
        for reservoir in reservoirs
    ]

    # Concatenate each augmented reservoir with input
    concatenations = [
        tf.keras.layers.Concatenate()([input_layer, exponentiation])
        for exponentiation in exponentiations
    ]

    # Create readout layer for each ensemble member
    readouts = []
    for i, concatenation in enumerate(concatenations):
        readout_config_copy = (
            readout_config.copy()
            if isinstance(readout_config, dict)
            else readout_config
        )
        if isinstance(readout_config_copy, dict):
            readout_config_copy["name"] = f"readout_{i}"
            readout_config_copy["units"] = features
        readout = ReadOut_builder(readout_config_copy)(concatenation)
        readouts.append(readout)

    # Combine outputs with outlier-filtered mean
    filtered_mean = OutliersFilteredMean()(readouts)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=filtered_mean, name=name)
    return model


def residual_stacked_ESN(
    units: Union[int, List[int]],
    reservoir_config: Union[
        str, List[str], Dict[str, Any], List[Dict[str, Any]]
    ] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[
        str, List[str], Dict[str, Any], List[Dict[str, Any]]
    ] = READOUT_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "stacked_ESN",
) -> tf.keras.Model:
    """
    Build a multi-layer ESN with residual connections.

    This model stacks multiple reservoir layers with residual-like connections,
    where each layer receives input from previous layers.

    Parameters
    ----------
    units : Union[int, list[int]]
        Number of units in each reservoir layer. If a single int is provided,
        a single-layer model will be created. If a list is provided, a multi-layer
        model will be created with the units specified in the list.
    reservoir_config : Union[str, List[str], Dict, List[Dict]], optional
        Configuration for each reservoir layer. Can be a single config or list of configs.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, List[str], Dict, List[Dict]], optional
        Configuration for readout layers. Can be a single config or list of configs.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.
    name : str, optional
        Name for the model. Default is "stacked_ESN".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the stacked reservoir model.

    Notes
    -----
    - The connectivity pattern is:
      - 1st layer: connected to input
      - 2nd layer: connected to input and 1st layer
      - 3rd+ layers: connected to the previous two layers
    """
    # Convert single configs to lists for consistency
    # Handle reservoir_config
    if isinstance(reservoir_config, (str, dict)):
        reservoir_configs = [
            (
                load_user_config(reservoir_config)
                if isinstance(reservoir_config, str)
                else reservoir_config.copy()
            )
        ]
    else:
        # Process list of configs
        reservoir_configs = []
        for config in reservoir_config:
            if isinstance(config, str):
                reservoir_configs.append(load_user_config(config))
            else:
                reservoir_configs.append(config.copy())

    # Handle readout_config
    if isinstance(readout_config, (str, dict)):
        readout_configs = [
            (
                load_user_config(readout_config)
                if isinstance(readout_config, str)
                else readout_config.copy()
            )
        ]
    else:
        # Process list of configs
        readout_configs = []
        for config in readout_config:
            if isinstance(config, str):
                readout_configs.append(load_user_config(config))
            else:
                readout_configs.append(config.copy())

    # Convert single units to list
    units_list = [units] if isinstance(units, int) else units

    # Validate configuration lengths
    if len(units_list) != len(reservoir_configs):
        # If we have one reservoir config but multiple units, replicate the config
        if len(reservoir_configs) == 1:
            reservoir_configs = reservoir_configs * len(units_list)
        else:
            raise ValueError(
                f"Number of units ({len(units_list)}) must match number of reservoir configs ({len(reservoir_configs)})"
            )

    # Configure dimensionality for each layer
    for i, (config, unit_n) in enumerate(zip(reservoir_configs, units_list)):
        config["units"] = unit_n

        # Set appropriate feedback dimensions based on layer position
        if i == 0:
            config["feedback_dim"] = features  # First layer takes raw input
        elif i == 1:
            config["feedback_dim"] = (
                units_list[i - 1] + features
            )  # Second layer takes input + first layer output
        else:
            config["feedback_dim"] = (
                units_list[i - 1] + units_list[i - 2]
            )  # Later layers take previous two layer outputs

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    # Build the stacked network with residual connections
    reservoirs = []
    for i, config in enumerate(reservoir_configs):
        if i == 0:
            # First layer connects to input only
            reservoir = ESNReservoir_builder(config)(input_layer)
        elif i == 1:
            # Second layer connects to input and first reservoir
            reservoir = ESNReservoir_builder(config)(
                tf.keras.layers.Concatenate()([input_layer, reservoirs[-1]])
            )
        else:
            # Later layers connect to previous two reservoirs
            reservoir = ESNReservoir_builder(config)(
                tf.keras.layers.Concatenate()([reservoirs[-1], reservoirs[-2]])
            )

        reservoirs.append(reservoir)

    # Configure and create readout layer
    readout_configs[0]["name"] = "readout"
    readout_configs[0]["units"] = features
    readout = ReadOut_builder(readout_configs[0])(reservoirs[-1])

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=readout, name=name)
    return model

def headless_ESN(
    units: int,
    reservoir_config: Union[str, Dict[str, Any]] = ESN_RESERVOIR_CONFIG,
    batch: int = 1,
    features: int = 1,
    name: str = "headless_ESN",
) -> tf.keras.Model:
    """
    Build an ESN model with no readout layer.
    
    This model can be used to study the dynamics of the reservoir by applying different transformations to the reservoir states.
    """
    # Load config from file if string is provided
    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    # Build reservoir
    overrides = {"units": units, "feedback_dim": features}
    reservoir = ESNReservoir_builder(reservoir_config, overrides=overrides)(input_layer)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=reservoir, name=name)
    return model
