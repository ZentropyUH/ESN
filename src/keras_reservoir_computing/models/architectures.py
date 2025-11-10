"""Pre-built architectures for reservoir computing models.

This module provides complete model architectures for different types of
Echo State Networks and other reservoir computing approaches. These architectures
are designed to be flexible, allowing users to customize layer configurations
through configuration dictionaries while maintaining sensible defaults.
"""

import os
from typing import Any, Dict, Optional, Union

import tensorflow as tf

from keras_reservoir_computing.io.config_models import (
    ReadoutConfig,
    ReservoirConfig,
)
from keras_reservoir_computing.io.loaders import (
    load_object,
    load_readout_config,
    load_reservoir_config,
)
from keras_reservoir_computing.layers import (
    OutliersFilteredMean,
    SelectiveExponentiation,
)


def classic_ESN(
    units: Optional[int] = None,
    reservoir_config: Optional[Union[str, Dict[str, Any], ReservoirConfig]] = None,
    readout_config: Optional[Union[str, Dict[str, Any], ReadoutConfig]] = None,
    batch: int = 1,
    features: int = 1,
    name: str = "classic_ESN",
) -> tf.keras.Model:
    """Build a classic Echo State Network (ESN) model.

    This function creates a classic ESN architecture where the input is
    concatenated with the reservoir output before passing to the readout layer.

    Parameters
    ----------
    units : int, optional
        Number of units in the reservoir. If provided, this will be used only
        if ``units`` is not already specified in ``reservoir_config``.
        If neither is provided, the default from the config will be used.
    reservoir_config : Union[str, Dict[str, Any], ReservoirConfig], optional
        Configuration for the reservoir layer. Can be:
        - A path to a YAML/JSON file
        - A dictionary with configuration
        - A ReservoirConfig instance
        - None to use default configuration

        The configuration can specify all reservoir parameters including
        ``units``, ``activation``, ``initializers``, etc. Parameters specified
        here take precedence over function parameters. Note that ``feedback_dim``
        is always set to match ``features`` as an architectural requirement
        (the input serves as feedback for these architectures).
    readout_config : Union[str, Dict[str, Any], ReadoutConfig], optional
        Configuration for the readout layer. Can be:
        - A path to a YAML/JSON file
        - A dictionary with configuration
        - A ReadoutConfig instance
        - None to use default configuration

        The configuration should specify all readout parameters including
        ``units``, ``alpha`` (for RidgeReadout), etc.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. This determines the input shape
        and is used to set ``feedback_dim`` in the reservoir config
        (architecture-specific requirement). Default is 1.
    name : str, optional
        Name for the model. Default is "classic_ESN".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the classic ESN.

    Notes
    -----
    The architecture is: Input Layer -> Reservoir -> Concatenate -> Readout Layer
                                     -------------->
    
    Layer-specific parameters (activation, initializers, etc.) should be
    specified in the configuration dictionaries. The architecture function
    only sets architecture-specific parameters (like ``feedback_dim = features``)
    when necessary. Config values are never overridden by function parameters
    unless they are architecture-specific requirements.

    Examples
    --------
    >>> import keras_reservoir_computing as krc
    >>> # Quick creation with units parameter
    >>> model = krc.classic_ESN(units=100, features=2, batch=1)
    >>>
    >>> # Use custom config with units specified (units parameter ignored)
    >>> reservoir_config = {
    ...     "class_name": "krc>ESNReservoir",
    ...     "config": {
    ...         "units": 200,  # This takes precedence
    ...         "activation": "tanh",
    ...         "leak_rate": 0.9
    ...     }
    ... }
    >>> model = krc.classic_ESN(
    ...     units=100,  # Ignored since units is in config
    ...     reservoir_config=reservoir_config,
    ...     features=2
    ... )
    """
    # Load and validate configurations
    reservoir_cfg = load_reservoir_config(reservoir_config)
    readout_cfg = load_readout_config(readout_config)

    # Get dtype from reservoir config, or default to float32
    dtype = reservoir_cfg.config.get("dtype", "float32")

    # Set architecture-specific parameters
    # feedback_dim must match features for this architecture (always override)
    reservoir_cfg = reservoir_cfg.override_config(feedback_dim=features)

    # Set units if provided and not in config (convenience parameter)
    if units is not None and "units" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(units=units)

    # Ensure dtype consistency (set if not present)
    if "dtype" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(dtype=dtype)
    if "dtype" not in readout_cfg.config:
        readout_cfg = readout_cfg.update_config(dtype=dtype)

    # Create the input layer
    input_layer = tf.keras.layers.Input(
        shape=(None, features), batch_size=batch, dtype=dtype
    )

    # Create reservoir layer
    reservoir = load_object(reservoir_cfg)(input_layer)

    # Concatenate input with reservoir output
    concat = tf.keras.layers.Concatenate(dtype=dtype)([input_layer, reservoir])

    # Create readout layer
    readout = load_object(readout_cfg)(concat)

    # Build and return model
    model = tf.keras.Model(
        inputs=input_layer, outputs=readout, name=name, dtype=dtype
    )
    return model


def Ott_ESN(
    units: Optional[int] = None,
    reservoir_config: Optional[Union[str, Dict[str, Any], ReservoirConfig]] = None,
    readout_config: Optional[Union[str, Dict[str, Any], ReadoutConfig]] = None,
    batch: int = 1,
    features: int = 1,
    name: str = "Ott_ESN",
) -> tf.keras.Model:
    """Build Ott's ESN model with state augmentation.

    This model follows the architecture proposed by Edward Ott, which augments
    reservoir states by squaring even-indexed units and concatenating with input.
    This augmentation helps capture higher-order dynamics in the reservoir states.

    Parameters
    ----------
    units : int, optional
        Number of units in the reservoir. If provided, this will be used only
        if ``units`` is not already specified in ``reservoir_config``.
        If neither is provided, the default from the config will be used.
    reservoir_config : Union[str, Dict[str, Any], ReservoirConfig], optional
        Configuration for the reservoir layer. Can be:
        - A path to a YAML/JSON file
        - A dictionary with configuration
        - A ReservoirConfig instance
        - None to use default configuration

        The configuration can specify all reservoir parameters including
        ``units``, ``feedback_dim``, ``activation``, ``initializers``, etc.
        Parameters specified here take precedence over function parameters.
        The ``feedback_dim`` will be automatically set to match ``features``
        (architecture-specific requirement).
    readout_config : Union[str, Dict[str, Any], ReadoutConfig], optional
        Configuration for the readout layer. Can be:
        - A path to a YAML/JSON file
        - A dictionary with configuration
        - A ReadoutConfig instance
        - None to use default configuration

        The configuration should specify all readout parameters including
        ``units``, ``alpha`` (for RidgeReadout), etc.
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. This will be used to set
        ``feedback_dim`` in the reservoir config (architecture-specific
        requirement). Default is 1.
    name : str, optional
        Name for the model. Default is "Ott_ESN".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing Ott's ESN.

    Notes
    -----
    The architecture augments the reservoir output with squared values of
    even-indexed units. The augmented reservoir output is concatenated with
    the input before the readout layer.

    Architecture: Input -> Reservoir -> SelectiveExponentiation -> Concatenate -> Readout

    Layer-specific parameters (activation, initializers, etc.) should be
    specified in the configuration dictionaries. Config values are never
    overridden by function parameters unless they are architecture-specific
    requirements.

    References
    ----------
    .. [1] E. Ott, J. Pathak, B. Hunt, M. Girvan, and Z. Lu, "Model-Free
       Prediction of Large Spatiotemporally Chaotic Systems from Data: A
       Reservoir Computing Approach," Phys. Rev. Lett., vol. 120, no. 2,
       p. 024102, Jan. 2018.

    Examples
    --------
    >>> import keras_reservoir_computing as krc
    >>> # Quick creation with units parameter
    >>> model = krc.Ott_ESN(units=200, features=2, batch=1)
    >>>
    >>> # Use custom reservoir config
    >>> reservoir_config = {
    ...     "class_name": "krc>ESNReservoir",
    ...     "config": {
    ...         "units": 200,
    ...         "activation": "tanh",
    ...         "leak_rate": 0.95
    ...     }
    ... }
    >>> model = krc.Ott_ESN(reservoir_config=reservoir_config, features=2)
    """
    # Load and validate configurations
    reservoir_cfg = load_reservoir_config(reservoir_config)
    readout_cfg = load_readout_config(readout_config)

    # Get dtype from reservoir config, or default to float32
    dtype = reservoir_cfg.config.get("dtype", "float32")

    # Set architecture-specific parameters
    # feedback_dim must match features for this architecture (always override)
    reservoir_cfg = reservoir_cfg.override_config(feedback_dim=features)

    # Set units if provided and not in config (convenience parameter)
    if units is not None and "units" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(units=units)

    # Ensure dtype consistency (set if not present)
    if "dtype" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(dtype=dtype)
    if "dtype" not in readout_cfg.config:
        readout_cfg = readout_cfg.update_config(dtype=dtype)

    # Create input layer
    feedback_layer = tf.keras.layers.Input(
        shape=(None, features), batch_size=batch, dtype=dtype
    )

    # Create reservoir layer
    reservoir = load_object(reservoir_cfg)(feedback_layer)

    # Augment reservoir output by squaring even-indexed units
    selective_exponentiation = SelectiveExponentiation(
        index=0, exponent=2.0, dtype=dtype
    )(reservoir)

    # Concatenate original input with augmented reservoir output
    concat = tf.keras.layers.Concatenate(dtype=dtype)(
        [feedback_layer, selective_exponentiation]
    )

    # Create readout layer
    readout = load_object(readout_cfg)(concat)

    # Build and return model
    model = tf.keras.Model(
        inputs=feedback_layer, outputs=readout, name=name, dtype=dtype
    )
    return model


def headless_ESN(
    units: Optional[int] = None,
    reservoir_config: Optional[Union[str, Dict[str, Any], ReservoirConfig]] = None,
    batch: int = 1,
    features: int = 1,
    name: str = "headless_ESN",
) -> tf.keras.Model:
    """Build an ESN model with no readout layer.

    This model can be used to study the dynamics of the reservoir by applying
    different transformations to the reservoir states without a readout layer.
    Useful for analyzing reservoir dynamics, state space properties, and
    feature extraction.

    Parameters
    ----------
    units : int, optional
        Number of units in the reservoir. If provided, this will be used only
        if ``units`` is not already specified in ``reservoir_config``.
        If neither is provided, the default from the config will be used.
    reservoir_config : Union[str, Dict[str, Any], ReservoirConfig], optional
        Configuration for the reservoir layer. Can be:
        - A path to a YAML/JSON file
        - A dictionary with configuration
        - A ReservoirConfig instance
        - None to use default configuration

        The configuration can specify all reservoir parameters including
        ``units``, ``feedback_dim``, ``activation``, ``initializers``, etc.
        Parameters specified here take precedence over function parameters.
        The ``feedback_dim`` will be automatically set to match ``features``
        (architecture-specific requirement).
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. This will be used to set
        ``feedback_dim`` in the reservoir config (architecture-specific
        requirement). Default is 1.
    name : str, optional
        Name for the model. Default is "headless_ESN".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the headless ESN (reservoir only).

    Notes
    -----
    The architecture is: Input Layer -> Reservoir

    The reservoir is not connected to a readout layer, allowing direct
    access to reservoir states for analysis or custom processing.

    Layer-specific parameters (activation, initializers, etc.) should be
    specified in the configuration dictionary. Config values are never
    overridden by function parameters unless they are architecture-specific
    requirements.

    Examples
    --------
    >>> import keras_reservoir_computing as krc
    >>> # Quick creation with units parameter
    >>> model = krc.headless_ESN(units=100, features=2, batch=1)
    >>>
    >>> # Use custom reservoir config
    >>> reservoir_config = {
    ...     "class_name": "krc>ESNReservoir",
    ...     "config": {
    ...         "units": 100,
    ...         "activation": "tanh",
    ...         "leak_rate": 0.9
    ...     }
    ... }
    >>> model = krc.headless_ESN(reservoir_config=reservoir_config, features=2)
    """
    # Load and validate configuration
    reservoir_cfg = load_reservoir_config(reservoir_config)

    # Get dtype from reservoir config, or default to float32
    dtype = reservoir_cfg.config.get("dtype", "float32")

    # Set architecture-specific parameters
    # feedback_dim must match features for this architecture (always override)
    reservoir_cfg = reservoir_cfg.override_config(feedback_dim=features)

    # Set units if provided and not in config (convenience parameter)
    if units is not None and "units" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(units=units)

    # Ensure dtype is set (set if not present)
    if "dtype" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(dtype=dtype)

    # Create input layer
    input_layer = tf.keras.layers.Input(
        shape=(None, features), batch_size=batch, dtype=dtype
    )

    # Create reservoir layer
    reservoir = load_object(reservoir_cfg)(input_layer)

    # Build and return model
    model = tf.keras.Model(
        inputs=input_layer, outputs=reservoir, name=name, dtype=dtype
    )
    return model


def linear_ESN(
    units: Optional[int] = None,
    reservoir_config: Optional[Union[str, Dict[str, Any], ReservoirConfig]] = None,
    batch: int = 1,
    features: int = 1,
    name: str = "linear_ESN",
) -> tf.keras.Model:
    """Build an ESN model with no readout layer and a linear activation function.

    This model uses a linear activation function in the reservoir, which can be
    useful for studying linear dynamics or as a baseline for comparison with
    nonlinear reservoirs.

    Parameters
    ----------
    units : int, optional
        Number of units in the reservoir. If provided, this will be used only
        if ``units`` is not already specified in ``reservoir_config``.
        If neither is provided, the default from the config will be used.
    reservoir_config : Union[str, Dict[str, Any], ReservoirConfig], optional
        Configuration for the reservoir layer. Can be:
        - A path to a YAML/JSON file
        - A dictionary with configuration
        - A ReservoirConfig instance
        - None to use default configuration

        The configuration can specify all reservoir parameters including
        ``units``, ``feedback_dim``, ``initializers``, etc. Parameters
        specified here take precedence over function parameters (except
        ``activation``, which is always set to "linear" for this architecture).
        The ``feedback_dim`` will be automatically set to match ``features``
        (architecture-specific requirement).
    batch : int, optional
        Batch size for the input layer. Default is 1.
    features : int, optional
        Number of features in the input data. This will be used to set
        ``feedback_dim`` in the reservoir config (architecture-specific
        requirement). Default is 1.
    name : str, optional
        Name for the model. Default is "linear_ESN".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the linear ESN (reservoir with linear activation).

    Notes
    -----
    The architecture is: Input Layer -> Reservoir (with linear activation)

    The reservoir uses a linear activation function, making it equivalent to
    a linear dynamical system. This can be useful for theoretical analysis
    or as a baseline model.

    Layer-specific parameters (initializers, etc.) should be specified in the
    configuration dictionary. The activation is always set to "linear" as an
    architecture-specific requirement for this model.

    Examples
    --------
    >>> import keras_reservoir_computing as krc
    >>> # Quick creation with units parameter
    >>> model = krc.linear_ESN(units=100, features=2, batch=1)
    >>>
    >>> # Use custom reservoir config (activation will be set to linear)
    >>> reservoir_config = {
    ...     "class_name": "krc>ESNReservoir",
    ...     "config": {
    ...         "units": 100,
    ...         "leak_rate": 0.9
    ...     }
    ... }
    >>> model = krc.linear_ESN(reservoir_config=reservoir_config, features=2)
    """
    # Load and validate configuration
    reservoir_cfg = load_reservoir_config(reservoir_config)

    # Get dtype from reservoir config, or default to float32
    dtype = reservoir_cfg.config.get("dtype", "float32")

    # Set architecture-specific parameters
    # feedback_dim must match features for this architecture (always override)
    reservoir_cfg = reservoir_cfg.override_config(feedback_dim=features)

    # Set linear activation (this is the key difference from headless_ESN)
    # This is an architecture-specific requirement, so we override
    reservoir_cfg = reservoir_cfg.override_config(activation="linear")

    # Set units if provided and not in config (convenience parameter)
    if units is not None and "units" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(units=units)

    # Ensure dtype is set (set if not present)
    if "dtype" not in reservoir_cfg.config:
        reservoir_cfg = reservoir_cfg.update_config(dtype=dtype)

    # Create input layer
    input_layer = tf.keras.layers.Input(
        shape=(None, features), batch_size=batch, dtype=dtype
    )

    # Create reservoir layer
    reservoir = load_object(reservoir_cfg)(input_layer)

    # Build and return model
    model = tf.keras.Model(
        inputs=input_layer, outputs=reservoir, name=name, dtype=dtype
    )
    return model


def ensemble_model(
    models_dir: str,
    n_models: int,
    method: str = "z_score",
    threshold: float = 5.0,
    dtype: str = "float32",
) -> tf.keras.Model:
    """Load multiple models from a directory and build an ensemble model.

    This function loads all `.keras` models from a directory and combines them
    into an ensemble model. Each submodel receives the same input tensor, and
    their outputs are merged using the OutliersFilteredMean layer, which filters
    outlier predictions before computing the mean.

    Parameters
    ----------
    models_dir : str
        Directory path containing the `.keras` model files.
    n_models : int
        Maximum number of models to load from the directory. Models are loaded
        in alphabetical order.
    method : str, optional
        Method for outlier filtering. Can be "z_score" or "iqr".
        Default is "z_score".
    threshold : float, optional
        Threshold for outlier detection. Default is 5.0.
    dtype : str, optional
        Data type for the model. Default is "float32".

    Returns
    -------
    tf.keras.Model
        A Keras Model representing the ensemble of loaded models.

    Raises
    ------
    ValueError
        If no `.keras` models are found in the directory.

    Notes
    -----
    The ensemble model combines predictions from multiple models by filtering
    outliers and computing the mean of the remaining predictions. This can
    improve robustness and reduce the impact of individual model errors.

    Examples
    --------
    >>> import keras_reservoir_computing as krc
    >>> # Load ensemble from directory
    >>> model = krc.ensemble_model(
    ...     models_dir="./models",
    ...     n_models=5,
    ...     method="z_score",
    ...     threshold=3.0
    ... )
    """
    files = sorted(f for f in os.listdir(models_dir) if f.endswith(".keras"))[
        :n_models
    ]
    if not files:
        raise ValueError(f"No .keras models found in {models_dir!r}")

    loaded = [
        tf.keras.models.load_model(os.path.join(models_dir, f), compile=False)
        for f in files
    ]

    input_shape = loaded[0].input_shape[1:]
    x = tf.keras.Input(shape=input_shape, name="ensemble_input", dtype=dtype)

    outputs = []
    for i, model in enumerate(loaded):
        model.name = model.name + f"__{i}"
        outputs.append(model(x))

    y = OutliersFilteredMean(
        name="merge_out", method=method, threshold=threshold, dtype=dtype
    )(outputs)
    return tf.keras.Model(inputs=x, outputs=y, name="ensemble_model")
