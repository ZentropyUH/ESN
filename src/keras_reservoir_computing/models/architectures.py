import tensorflow as tf
from typing import List, Union

from keras_reservoir_computing.layers import (
    OutliersFilteredMean,
    SelectiveExponentiation,
)
from keras_reservoir_computing.layers.builders import (
    ESNReservoir_builder,
    ReadOut_builder,
)
from keras_reservoir_computing.layers.config import load_user_config

ESN_RESERVOIR_CONFIG = {
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


READOUT_CONFIG = {
    "kind": "ridge",
    "units": 100,
    "alpha": 0.1,
    "trainable": False,
}

def classic_ESN(
    units, 
    reservoir_config: Union[str, dict] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, dict] = READOUT_CONFIG,
    batch=1,
    features=1,
    name="classic_ESN",
) -> tf.keras.Model:
    """
    Builds and returns a classic Echo State Network (ESN) using Keras.

    Parameters
    ----------
    units : int
        Number of units in the reservoir.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a string identifier or a dictionary of parameters.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, dict], optional
        Configuration for the readout layer. Can be a string identifier or a dictionary of parameters.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.

    Returns
    -------
    keras.Model
        A Keras Model representing the classic ESN.

    Notes
    -----
    - The model built is based on the classic ESN formulation.
    - The architecture is the following:
        - `Input Layer` -> `ESN Reservoir` -> `Readout Layer`

    """

    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    overrides = {"units": units}
    reservoir = ESNReservoir_builder(reservoir_config, overrides=overrides)(input_layer)

    readout = ReadOut_builder(readout_config)(reservoir)

    model = tf.keras.Model(inputs=input_layer, outputs=readout, name=name)
    return model

# Simple ESN With no input signal, only feedback
def Ott_ESN(
    units,
    reservoir_config: Union[str, dict] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, dict] = READOUT_CONFIG,
    batch=1,
    features=1,
    name="Ott_ESN",
) -> tf.keras.Model:
    """
    Builds and returns Ott's proposed model with state augmentation, feedback-only Echo State Network (ESN) using Keras.

    Parameters
    ----------
    units : int
        Number of units in the reservoir.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a string identifier or a dictionary of parameters.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, dict], optional
        Configuration for the readout layer. Can be a string identifier or a dictionary of parameters.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.

    Returns
    -------
    keras.Model
        A Keras Model representing Ott's ESN.

    Notes
    -----
    - The model built is based on Edward Ott's ESN formulation.
    - The architecture is the following:
        - `Input Layer` -> `ESN Reservoir` -> `Readout Layer`
        - The caveat is to augment the reservoir output with a selective exponentiation of even positions to their squared value, afterwards concatenate the augmented reservoir output with the input (feedback), then feed it to the readout layer.

    References
    ----------
    .. E. Ott, J. Pathak, B. Hunt, M. Girvan, and Z. Lu, “Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach,” Phys. Rev. Lett., vol. 120, no. 2, p. 024102, Jan. 2018, doi: 10.1103/PhysRevLett.120.024102.


    """

    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    feedback_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    overrides = {"units": units, "feedback_dim": features}
    reservoir = ESNReservoir_builder(reservoir_config, overrides=overrides)(
        feedback_layer
    )

    selective_exponentiation = SelectiveExponentiation(
        index=0,
        exponent=2.0,
    )(reservoir)

    concat = tf.keras.layers.Concatenate()([feedback_layer, selective_exponentiation])

    readout = ReadOut_builder(readout_config)(concat)

    model = tf.keras.Model(inputs=feedback_layer, outputs=readout, name=name)
    return model


def ensemble_with_mean_ESN(
    units: int,
    ensemble_size: int,
    reservoir_config: Union[str, dict] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, dict] = READOUT_CONFIG,
    batch=1,
    features=1,
    name="ensemble_with_mean_ESN",
) -> tf.keras.Model:

    if isinstance(reservoir_config, str):
        reservoir_config = load_user_config(reservoir_config)

    if isinstance(readout_config, str):
        readout_config = load_user_config(readout_config)

    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    reservoir_config["units"] = units
    reservoirs = [
        ESNReservoir_builder(reservoir_config)(input_layer)
        for _ in range(ensemble_size)
    ]

    exponentiations = [
        SelectiveExponentiation(
            index=0,
            exponent=2.0,
        )(reservoir)
        for reservoir in reservoirs
    ]

    concatenations = [
        tf.keras.layers.Concatenate()([input_layer, exponentiation])
        for exponentiation in exponentiations
    ]

    readouts = []

    for i, concatenation in enumerate(concatenations):
        readout_config["name"] = f"readout_{i}"
        readout = ReadOut_builder(readout_config)(concatenation)
        readouts.append(readout)

    # All readouts are of shape (batch, timestep, features). We need to stack them along a new axis on the left, so the sahpe becomes (ensemble_size, batch, timestep, features)

    filtered_mean = OutliersFilteredMean()(readouts)

    model = tf.keras.Model(inputs=input_layer, outputs=filtered_mean, name=name)
    return model


def residual_stacked_ESN(
    units: Union[int, list[int]],
    reservoir_config: Union[str, List[str], dict, List[dict]] = ESN_RESERVOIR_CONFIG,
    readout_config: Union[str, List[str], dict, List[dict]] = READOUT_CONFIG,
    batch=1,
    features=1,
    name="stacked_ESN",
) -> tf.keras.Model:
    """
    Builds and returns a stacked reservoir model using Keras.

    Parameters
    ----------
    units : Union[int, list[int]]
        Number of units in the reservoir. If a list is provided, it will build a stacked reservoir model with the number of units specified in the list.
    reservoir_config : Union[str, List[str], dict, List[dict]], optional
        Configuration for the reservoir. Can be a string identifier, a list of string identifiers, a dictionary of parameters, or a list of dictionaries of parameters.
        Default is ESN_RESERVOIR_CONFIG.
    readout_config : Union[str, List[str], dict, List[dict]], optional
        Configuration for the readout layer. Can be a string identifier, a list of string identifiers, a dictionary of parameters, or a list of dictionaries of parameters.
        Default is READOUT_CONFIG.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.

    Returns
    -------
    keras.Model
        A Keras Model representing the stacked reservoir model.

    Notes
    -----
    - The model built is based on Edward Ott's ESN formulation.
    - The architecture is the following:
        - `Input Layer` -> `ESN Reservoir` -> `Readout Layer`
        - The caveat is to augment the reservoir output with a selective exponentiation of even positions to their squared value, afterwards concatenate the augmented reservoir output with the input (feedback), then feed it to the readout layer.

    """

    if isinstance(reservoir_config, str):
        reservoir_config = [load_user_config(reservoir_config)]

    elif isinstance(reservoir_config, list):
        if not all(isinstance(config, (str, dict)) for config in reservoir_config):
            raise ValueError(
                "All elements in reservoir_config must be either string identifiers or dictionaries."
            )
        if all(isinstance(config, str) for config in reservoir_config):
            reservoir_config = [load_user_config(config) for config in reservoir_config]

    if isinstance(readout_config, str):
        readout_config = [load_user_config(readout_config)]

    elif isinstance(readout_config, list):
        if not all(isinstance(config, (str, dict)) for config in readout_config):
            raise ValueError(
                "All elements in readout_config must be either string identifiers or dictionaries."
            )
        if all(isinstance(config, str) for config in readout_config):
            readout_config = [load_user_config(config) for config in readout_config]

    if isinstance(units, int):
        units = [units]

    if len(units) != len(reservoir_config):
        raise ValueError("Number of units and reservoir configurations must match.")

    # Override the units in the reservoir configurations
    for i, (config, unit_n) in enumerate(zip(reservoir_config, units)):
        config["units"] = unit_n
        if i == 0:
            config["feedback_dim"] = features
        if i == 1:
            config["feedback_dim"] = units[i - 1] + features
        if i > 1:
            config["feedback_dim"] = units[i - 1] + units[i - 2]

    # we are going to make like a residual network of the stacked reservoirs, the first reservoir will be connected to the input layer, the second reservoir will be connected to the concatenation of the input layer and the first reservoir, then from the third reservoir, it will be connected to the concatenation of the previous two reservoirs.

    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch)

    reservoirs = []

    for i, config in enumerate(reservoir_config):
        if i == 0:
            reservoir = ESNReservoir_builder(config)(input_layer)
        elif i == 1:
            reservoir = ESNReservoir_builder(config)(
                tf.keras.layers.Concatenate()([input_layer, reservoirs[-1]])
            )
        else:
            reservoir = ESNReservoir_builder(config)(
                tf.keras.layers.Concatenate()([reservoirs[-1], reservoirs[-2]])
            )

        reservoirs.append(reservoir)

    readout_config[0]["name"] = "readout"

    readout = ReadOut_builder(readout_config[0])(reservoirs[-1])

    model = tf.keras.Model(inputs=input_layer, outputs=readout, name=name)
    return model
