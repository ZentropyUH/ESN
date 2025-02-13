import inspect
import json
import types
from typing import Tuple, Optional, Dict, Any


def get_default_params(cls: type) -> Dict[str, Any]:
    """
    Retrieve the default parameters from a class's __init__ method.

    This function inspects the signature of the class constructor to extract
    argument names and their default values. Parameters without default values
    (including ``self``) are excluded.

    Parameters
    ----------
    cls : type
        The class from which to extract default parameters.

    Returns
    -------
    dict
        A dictionary mapping parameter names to their default values.
        Only parameters that have default values are included.

    Raises
    ------
    ValueError
        If the class has no __init__ method or it cannot be inspected.
    """
    # Get the signature of the class constructor
    signature = inspect.signature(cls.__init__)
    # Create a dictionary of parameters that have default values
    return {
        param.name: param.default
        for param in signature.parameters.values()
        if param.default is not param.empty and param.name != "self"
    }


def merge_with_defaults(
    default_params: Dict[str, Any],
    user_params: Dict[str, Any],
    override_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge user-specified parameters with a dictionary of default parameters.

    The function creates a new dictionary, copying all default parameter values first,
    and then overriding them with values from ``user_params``. Finally, if
    ``override_params`` is provided, those values take precedence over both defaults
    and user-specified parameters.

    Parameters
    ----------
    default_params : dict
        Dictionary of default parameters for a class or function.
    user_params : dict
        User-provided parameters to override or extend the defaults.
    override_params : dict, optional
        Additional parameters that override both defaults and user_params if present.

    Returns
    -------
    dict
        The merged dictionary of parameters, where user-specified values override
        defaults, and ``override_params`` override everything else.
    """
    # Start with default parameters, override with user_params
    params = {
        key: user_params.get(key, default) for key, default in default_params.items()
    }
    # Optionally apply final overrides
    if override_params is not None:
        params.update(override_params)

    return params


def get_class_from_name(name: str, module: types.ModuleType) -> type:
    """
    Retrieve a class object from a given module by its name.

    Parameters
    ----------
    name : str
        The name of the class to retrieve.
    module : types.ModuleType
        The Python module from which to retrieve the class.

    Returns
    -------
    type
        The class object with the specified ``name``.

    Raises
    ------
    ValueError
        If no class with the given name is found in the module.
    """
    class_ = getattr(module, name, None)
    if class_ is None:
        raise ValueError(f"Unknown class: '{name}' from module '{module.__name__}'.")
    return class_


def load_config(filepath: str, keys: Tuple[str, ...]) -> Dict[str, Any]:
    """
    Load a configuration dictionary from a JSON file and validate required keys.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.
    keys : tuple of str
        Keys that the configuration file must contain.

    Returns
    -------
    dict
        A dictionary loaded from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If any of the required keys are missing from the loaded dictionary.
    """
    with open(filepath, "r") as file:
        config = json.load(file)

    # Validate that all required keys are present
    missing_keys = [k for k in keys if k not in config]
    if missing_keys:
        raise KeyError(
            f"The following keys are missing from '{filepath}': {missing_keys}. "
            f"Expected keys: {list(keys)}."
        )

    return config


def load_model_config(filepath: str) -> Dict[str, Any]:
    """
    Load a model configuration dictionary from a JSON file.

    The JSON file is expected to include the following keys:
    ``["feedback_init", "feedback_bias_init", "kernel_init", "cell"]``.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        The model configuration dictionary, guaranteed to have the keys
        specified above.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If any required keys are missing from the JSON.
    """
    model_config_keys = ("feedback_init", "feedback_bias_init", "kernel_init", "cell")
    model_config = load_config(filepath, model_config_keys)
    return model_config


def load_train_config(filepath: str) -> Dict[str, Any]:
    """
    Load a training configuration dictionary from a JSON file.

    The JSON file is expected to include the following keys:
    ``["init_transient_length", "train_length", "transient_length", "normalize", "regularization"]``.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        The training configuration dictionary, guaranteed to have the keys
        specified above.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If any required keys are missing from the JSON.
    """
    train_config_keys = (
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    )
    train_config = load_config(filepath, train_config_keys)
    return train_config


def load_forecast_config(filepath: str) -> Dict[str, Any]:
    """
    Load a forecast configuration dictionary from a JSON file.

    The JSON file is expected to include the following keys:
    ``["forecast_length", "internal_states"]``.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        The forecast configuration dictionary, guaranteed to have the keys
        specified above.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If any required keys are missing from the JSON.
    """
    forecast_config_keys = ("forecast_length", "internal_states")
    forecast_config = load_config(filepath, forecast_config_keys)
    return forecast_config
