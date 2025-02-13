import os
from typing import Optional, Union, Dict, Any

import numpy as np
from keras.src.initializers import RandomUniform
from keras.src.layers import Dense
from rich.progress import track

from keras_reservoir_computing import initializers
from keras_reservoir_computing.models import ReservoirComputer, ReservoirEnsemble
from keras_reservoir_computing.reservoirs import EchoStateNetwork, ESNCell
from keras_reservoir_computing.utils.data_utils import list_files_only
from keras_reservoir_computing.utils.general_utils import timer

from .config import (
    get_class_from_name,
    get_default_params,
    load_model_config,
    merge_with_defaults,
)


def load_model(filepath: str) -> ReservoirComputer:
    """
    Load a previously saved reservoir computing model from a file.

    Parameters
    ----------
    filepath : str
        Path to the model file (e.g., a file saved using
        ``ReservoirComputer.save()``).

    Returns
    -------
    ReservoirComputer
        The loaded reservoir computing model.

    Raises
    ------
    OSError
        If the file cannot be read or does not exist.
    ValueError
        If the file contents are not a valid ``ReservoirComputer`` representation.

    Notes
    -----
    Make sure that the file at ``filepath`` is a valid model file previously
    saved via the ``krc.models.ReservoirComputer.save`` method.
    """
    return ReservoirComputer.load(filepath)


def create_model(
    name: str,
    model_config: Union[str, Dict[str, Any]],
    features: int,
    seed: Optional[int] = None,
    log: bool = False,
) -> ReservoirComputer:
    """
    Create a reservoir computing model (``ReservoirComputer``) based on a
    provided configuration and number of output features.

    Parameters
    ----------
    name : str
        A human-readable name for the model.
    model_config : str or dict
        Either a path to a JSON configuration file or a dictionary containing
        the keys:

        - ``feedback_init``: Dict with "name" and "params" describing the matrix initializer.
        - ``feedback_bias_init``: Dict with parameters for Keras's
          :class:`RandomUniform` initializer.
        - ``kernel_init``: Dict with "name" and "params" describing the reservoir
          topology initializer.
        - ``cell``: Dict of parameters for the ESNCell (e.g., alpha, connectivity).

        Example structure of ``model_config`` if given as a dict:
        ::

            {
                "feedback_init": {
                    "name": "InputMatrix",
                    "params": {
                        "connectivity": 0.1,
                        ...
                    }
                },
                "feedback_bias_init": {
                    "minval": -0.05,
                    "maxval": 0.05,
                    ...
                },
                "kernel_init": {
                    "name": "WattsStrogatzGraphInitializer",
                    "params": {
                        "n": 1000,
                        "k": 10,
                        "p": 0.2,
                        ...
                    }
                },
                "cell": {
                    "units": 200,
                    "alpha": 0.9,
                    ...
                },
                "seed": 123  # optional
            }

    features : int
        The dimensionality of the output (size of the readout layer).
    seed : int, optional
        Global random seed to ensure reproducible weight initialization.
        Defaults to None, in which case it will use:
        1. The seed from ``model_config["seed"]`` if present
        2. Otherwise, a random integer from ``np.random.randint(0, 1000000)``.
    log : bool, optional
        If True, prints debugging/log information about component initialization.

    Returns
    -------
    ReservoirComputer
        A configured reservoir computing model, including a reservoir and readout.

    Raises
    ------
    FileNotFoundError
        If ``model_config`` is a file path that does not exist.
    KeyError
        If the ``model_config`` dict is missing required keys.
    ValueError
        If any specified initializer class is not found.

    Notes
    -----
    - The model's reservoir is built using an :class:`ESNCell`.
    - The readout layer is a Keras :class:`Dense` layer with ``trainable=False``
      and linear activation (no activation).
    - The random seeds for the reservoir cell, feedback matrix, kernel (internal
      reservoir topology), and feedback bias are all set to the same integer for
      reproducibility.

    Example
    -------
    .. code-block:: python

        model_cfg = {
            "feedback_init": {"name": "InputMatrix", "params": {"connectivity": 0.2}},
            "feedback_bias_init": {"minval": -0.01, "maxval": 0.01},
            "kernel_init": {"name": "WattsStrogatzGraphInitializer", "params": {"n": 200, "k": 6, "p": 0.3}},
            "cell": {"units": 200, "alpha": 0.8},
        }
        rc_model = create_model(name="MyRC", model_config=model_cfg, features=1, seed=42)
    """
    # Possibly load JSON config from a filepath
    if isinstance(model_config, str):
        model_config = load_model_config(filepath=model_config)

    # Decide on the seed
    seed = (
        seed
        if seed is not None
        else model_config.get("seed", np.random.randint(0, 1000000))
    )
    model_config["seed"] = seed  # Store in config for consistency

    def initialize_component(component_key: str, default_name: str, module) -> Any:
        """
        Helper function to instantiate an initializer or component class by name.

        Parameters
        ----------
        component_key : str
            The key in `model_config` for the initializer (e.g. "feedback_init").
        default_name : str
            A fallback class name if none is specified in the config.
        module : module
            The Python module in which to look up the class (e.g., `initializers`).

        Returns
        -------
        Any
            An instantiated object of the specified class with merged parameters.
        """
        component_name = model_config[component_key].get("name", default_name)
        component_class = get_class_from_name(component_name, module)

        user_params = model_config[component_key].get("params", {})
        default_params = get_default_params(component_class)

        merged_params = merge_with_defaults(default_params, user_params, {"seed": seed})

        if log:
            print(f"{component_key} name:", component_name)
            print(f"{component_key} params:", merged_params)

        return component_class(**merged_params)

    # Initialize feedback and kernel
    feedback_init = initialize_component(
        component_key="feedback_init",
        default_name="InputMatrix",
        module=initializers,
    )
    kernel_init = initialize_component(
        component_key="kernel_init",
        default_name="WattsStrogatzGraphInitializer",
        module=initializers,
    )

    # Initialize feedback bias using Keras's RandomUniform
    # Note: If the user didn't specify 'minval'/'maxval', Keras uses defaults.
    feedback_bias_init_kwargs = model_config["feedback_bias_init"].copy()
    feedback_bias_init_kwargs["seed"] = seed
    feedback_bias_init = RandomUniform(**feedback_bias_init_kwargs)

    # Build the reservoir cell
    cell_params = model_config["cell"]
    cell = ESNCell(
        **cell_params,
        input_initializer=feedback_init,
        input_bias_initializer=feedback_bias_init,
        kernel_initializer=kernel_init,
    )

    # Wrap the cell in an EchoStateNetwork reservoir
    reservoir = EchoStateNetwork(reservoir_cell=cell)

    # Create a non-trainable readout layer
    readout_layer = Dense(
        units=features, activation="linear", name="readout", trainable=False
    )

    # Construct the final ReservoirComputer model
    model = ReservoirComputer(
        reservoir=reservoir,
        readout=readout_layer,
        seed=seed,
        name=name,
    )

    return model


def create_ensemble(
    trained_models_folder_path: str,
    ensemble_name: str = "Reservoir_Ensemble",
    log: bool = False,
) -> ReservoirEnsemble:
    """
    Create an ensemble from multiple trained reservoir computing models.

    Parameters
    ----------
    trained_models_folder_path : str
        Path to a folder containing only valid model files. Each file should
        be loadable via :func:`load_model`.
    ensemble_name : str, optional
        Name to assign to the ensemble. Defaults to "Reservoir_Ensemble".
    log : bool, optional
        If True, logs the time taken to load models using a context manager.

    Returns
    -------
    ReservoirEnsemble
        An ensemble of loaded ``ReservoirComputer`` models, grouped under
        a single ensemble name.

    Raises
    ------
    OSError
        If any of the model files cannot be read.
    ValueError
        If any of the files are not valid ``ReservoirComputer`` representations.

    Notes
    -----
    Internally, each file in ``trained_models_folder_path`` is loaded
    via :func:`load_model`, and all loaded models are then composed
    into a :class:`ReservoirEnsemble`. The ensemble can be used to combine
    forecasts from multiple reservoir computing models.

    Example
    -------
    .. code-block:: python

        # Suppose 'my_models' folder has 3 saved .h5 reservoir models:
        ensemble = create_ensemble("/path/to/my_models", ensemble_name="AllMyModels")
        # Now 'ensemble' holds all 3 models in a single object.
    """
    model_files = list_files_only(trained_models_folder_path)
    ensemble_models = []

    with timer("Loading models", log=log):
        for model_file in track(model_files, description="Loading models"):
            full_path = os.path.join(trained_models_folder_path, model_file)
            rc_model = load_model(full_path)
            ensemble_models.append(rc_model)

    ensemble = ReservoirEnsemble(
        reservoir_computers=ensemble_models,
        name=ensemble_name,
    )
    return ensemble
