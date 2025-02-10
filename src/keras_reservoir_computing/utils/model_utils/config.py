import inspect
import json
from typing import Tuple, Optional
import types


def get_default_params(cls: type) -> dict:
    """
    Retrieves the default parameters from a class's __init__ method.

    Parameters
    ----------
    cls : type
        The class from which to extract default parameters.

    Returns
    -------
    dict
        A dictionary of parameter names and their default values.
    """
    signature = inspect.signature(cls.__init__)
    return {
        param.name: param.default
        for param in signature.parameters.values()
        if param.default is not param.empty and param.name != "self"
    }


def merge_with_defaults(default_params: dict, user_params: dict, override_params: Optional[dict]=None) -> dict:
    """
    Merges user-specified parameters with default parameters of a class.

    Parameters
    ----------
    default_params : dict
        Default parameters of a class.
    user_params : dict
        User-provided parameters.

    Returns
    -------
    dict
        Merged dictionary with user-specified values overriding defaults.
    """
    params = {
        key: user_params.get(key, default)
        for key, default in default_params.items()  # Only override the default value if the key is present in the user_params
    }
    
    if override_params is not None:
        params.update(override_params)

    return params


def get_class_from_name(name: str, module: types.ModuleType) -> type:
    """
    Retrieves a class from a module by its name.

    Parameters
    ----------
    name : str
        Name of the class to retrieve
    module : module
        Module from which to retrieve the class
        
    Returns
    -------
    type
        The class with the given name.
    """
    class_ = getattr(module, name, None)
    if class_ is None:
        raise ValueError(f"Unknown class: {name} from module {module.__name__}")
    return class_


def load_config(filepath: str, keys: Tuple) -> dict:
    """
    Loads a configuration dictionary from a JSON file.

    This is a helper function to load a configuration dictionary from a JSON file, namely model_config, train_config, and forecast_config.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.
    keys : Tuple[str]
        Tuple of keys that the configuration dictionary should contain.
    Returns
    -------
    dict
        The loaded configuration dictionary.
    """
    with open(filepath, "r") as file:
        config = json.load(file)

    all_keys = config.keys()

    for key in keys:
        if key not in all_keys:
            raise KeyError(
                f"Key {key} not found in the configuration file {filepath}. It should contain the following keys: {keys}"
            )

    return config


def load_model_config(filepath: str) -> dict:
    """
    Loads a model configuration dictionary from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        The loaded model configuration dictionary.
    """
    model_config_keys = ["feedback_init", "feedback_bias_init", "kernel_init", "cell"]
    model_config = load_config(filepath, model_config_keys)
    return model_config


def load_train_config(filepath: str) -> dict:
    """
    Loads a training configuration dictionary from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        The loaded training configuration dictionary.
    """
    train_config_keys = [
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    ]
    train_config = load_config(filepath, train_config_keys)
    return train_config


def load_forecast_config(filepath: str) -> dict:
    """
    Loads a forecast configuration dictionary from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        The loaded forecast configuration dictionary.
    """
    forecast_config_keys = ["forecast_length", "internal_states"]
    forecast_config = load_config(filepath, forecast_config_keys)
    return forecast_config
