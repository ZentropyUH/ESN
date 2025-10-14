"""Pre-built architectures for reservoir computing models.

This module provides complete model architectures for different types of
Echo State Networks and other reservoir computing approaches.
"""

import os
from typing import Any, Dict, Optional, Union

import tensorflow as tf

# Import only what's needed to avoid circular imports
from keras_reservoir_computing.io.loaders import (
    load_config,
    load_default_config,
    load_object,
)
from keras_reservoir_computing.layers import (
    OutliersFilteredMean,
    SelectiveExponentiation,
)


def classic_ESN(
    units: int,
    reservoir_config: Optional[Union[str, Dict[str, Any]]] = None,
    readout_config: Optional[Union[str, Dict[str, Any]]] = None,
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
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary. If None, a default reservoir will be used.
    readout_config : Union[str, dict], optional
        Configuration for the readout layer. Can be a path to a JSON file or a dictionary. If None, a default readout layer will be used.
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
    - The architecture is: Input Layer -> Reservoir -> Concatenate -> Readout Layer
                                       -------------->
    """
    # Load config from file if string is provided
    reservoir_config = load_config(reservoir_config) if reservoir_config else load_default_config("reservoir")

    readout_config = load_config(readout_config) if readout_config else load_default_config("readout")

    reservoir_config.setdefault("config", {})
    readout_config.setdefault("config", {})

    reservoir_config["config"] |= {"units": units, "feedback_dim": features, "dtype": dtype}
    readout_config["config"]   |= {"dtype": dtype}

    # Create the input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)


    reservoir = load_object(reservoir_config)(input_layer)


    # Concatenate input with reservoir
    concat = tf.keras.layers.Concatenate(dtype=dtype)([input_layer, reservoir])

    readout = load_object(readout_config)(concat)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=readout, name=name, dtype=dtype)
    return model


def Ott_ESN(
    units: int,
    reservoir_config: Optional[Union[str, Dict[str, Any]]] = None,
    readout_config: Optional[Union[str, Dict[str, Any]]] = None,
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
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary. If None, a default reservoir will be used.
    readout_config : Union[str, dict], optional
        Configuration for the readout layer. Can be a path to a JSON file or a dictionary. If None, a default readout layer will be used.
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
    reservoir_config = load_config(reservoir_config) if reservoir_config else load_default_config("reservoir")
    readout_config = load_config(readout_config) if readout_config else load_default_config("readout")

    # Create input layer
    feedback_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)

    reservoir_config.setdefault("config", {})
    readout_config.setdefault("config", {})

    reservoir_config["config"] |= {"units": units, "feedback_dim": features, "dtype": dtype}
    readout_config["config"]   |= {"dtype": dtype}

    reservoir = load_object(reservoir_config)(feedback_layer)

    # Augment reservoir output by squaring even-indexed units
    selective_exponentiation = SelectiveExponentiation(
        index=0,
        exponent=2.0,
        dtype=dtype,
    )(reservoir)

    # Concatenate original input with augmented reservoir output
    concat = tf.keras.layers.Concatenate(dtype=dtype)([feedback_layer, selective_exponentiation])

    # Create readout layer
    readout = load_object(readout_config)(concat)

    # Build and return model
    model = tf.keras.Model(inputs=feedback_layer, outputs=readout, name=name, dtype=dtype)
    return model


def headless_ESN(
    units: int,
    reservoir_config: Optional[Union[str, Dict[str, Any]]] = None,
    batch: int = 1,
    features: int = 1,
    name: str = "headless_ESN",
    dtype: str = "float32",
) -> tf.keras.Model:
    """
    Build an ESN model with no readout layer.

    This model can be used to study the dynamics of the reservoir by applying different transformations to the reservoir states.

    Parameters
    ----------
    units : int
        Number of units in the reservoir.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary. If None, a default reservoir will be used.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.
    name : str, optional
        Name for the model. Default is "headless_ESN".
    dtype : str, optional
        Data type for the model. Default is "float32".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the headless ESN.

    Notes
    -----
    - The architecture is: Input Layer -> Reservoir
    - The reservoir is not connected to a readout layer.
    """
    # Load config from file if string is provided

    reservoir_config = load_config(reservoir_config) if reservoir_config else load_default_config("reservoir")

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)

    # Build reservoir
    reservoir_config.setdefault("config", {})
    reservoir_config["config"] |= {"units": units, "feedback_dim": features, "dtype": dtype}

    reservoir = load_object(reservoir_config)(input_layer)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=reservoir, name=name, dtype=dtype)
    return model


def linear_ESN(
    units: int,
    reservoir_config: Optional[Union[str, Dict[str, Any]]] = None,
    batch: int = 1,
    features: int = 1,
    name: str = "headless_ESN",
    dtype: str = "float32",
) -> tf.keras.Model:
    """
    Build an ESN model with no readout layer and a linear activation function.

    This model can be used to study the dynamics of the reservoir by applying different transformations to the reservoir states.

    Parameters
    ----------
    units : int
        Number of units in the reservoir.
    reservoir_config : Union[str, dict], optional
        Configuration for the reservoir. Can be a path to a JSON file or a dictionary. If None, a default reservoir will be used.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. Default is 1.
    name : str, optional
        Name for the model. Default is "headless_ESN".
    dtype : str, optional
        Data type for the model. Default is "float32".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the headless ESN.

    Notes
    -----
    - The architecture is: Input Layer -> Reservoir
    - The reservoir is not connected to a readout layer.
    """
    # Load config from file if string is provided

    reservoir_config = load_config(reservoir_config) if reservoir_config else load_default_config("reservoir")

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(None, features), batch_size=batch, dtype=dtype)

    # Build reservoir
    reservoir_config.setdefault("config", {})
    reservoir_config["config"] |= {"activation": "linear", "units": units, "feedback_dim": features, "dtype": dtype}

    reservoir = load_object(reservoir_config)(input_layer)

    # Build and return model
    model = tf.keras.Model(inputs=input_layer, outputs=reservoir, name=name, dtype=dtype)
    return model


def ensemble_model(models_dir: str, n_models: int | None = None) -> tf.keras.Model:
    """
    Build a parallel-ensemble from .keras files in a directory.
    Shared input -> list of model outputs -> OutliersFilteredMean -> final output.

    Parameters
    ----------
    models_dir : str
        Directory containing `.keras` model files.
    n_models : int | None
        If provided, use only the first n models (sorted by filename). If None, use all.

    Returns
    -------
    tf.keras.Model
        The assembled ensemble model.

    Raises
    ------
    ValueError
        If no models found, shapes mismatch, or outputs are incompatible.
    """
    # 1) Collect model files
    files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    files.sort()
    if not files:
        raise ValueError(f"No .keras models found in: {models_dir}")

    if n_models is not None:
        if n_models <= 0:
            raise ValueError("n_models must be positive if provided.")
        files = files[:n_models]

    # 2) Load and freeze models
    models = []
    for fname in files:
        path = os.path.join(models_dir, fname)
        m = tf.keras.models.load_model(path, compile=False)
        models.append(m)

    # 3) Validate shapes (all inputs/outputs identical)
    def _io_shapes(m: tf.keras.Model):
        if len(m.inputs) != 1 or len(m.outputs) != 1:
            raise ValueError(f"Model {m.name} must have exactly one input and one output.")
        in_shape = tuple(m.inputs[0].shape.as_list())[1:]   # drop batch dim
        out_shape = tuple(m.outputs[0].shape.as_list())[1:] # drop batch dim
        return in_shape, out_shape

    in0, out0 = _io_shapes(models[0])
    for m in models[1:]:
        ini, outi = _io_shapes(m)
        if ini != in0:
            raise ValueError(f"Input shape mismatch: {ini} != {in0}")
        if outi != out0:
            raise ValueError(f"Output shape mismatch: {outi} != {out0}")

    # OutliersFilteredMean expects per-branch outputs shaped (batch, timesteps, features)
    if len(out0) != 3:
        raise ValueError(
            f"OutliersFilteredMean expects 3D outputs (batch,timesteps,features); got {out0}"
        )

    # 4) Build graph: shared Input -> parallel models -> list -> OutliersFilteredMean
    inputs = tf.keras.Input(shape=in0, name="shared_input")
    branch_outputs = [m(inputs) for m in models]  # list of (B,T,F)
    aggregated = OutliersFilteredMean()(branch_outputs)  # (B,T,F)

    # 5) Compose final model
    ensemble = tf.keras.Model(inputs=inputs, outputs=aggregated, name="parallel_ensemble_outlier_mean")
    return ensemble
