import inspect
import json
import types
from typing import Optional, Dict, Union


def load_user_config(config: Union[str, Dict]) -> Dict:
    """
    Load JSON config from a file path or take an existing dict.
    """
    if isinstance(config, str):
        with open(config, "r") as f:
            return json.load(f)
    elif isinstance(config, dict):
        return config
    else:
        raise ValueError("Config must be a dict or a path to a JSON file.")


def get_default_params(cls: type) -> Dict:
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
    default_params: Dict,
    user_params: Dict,
    override_params: Optional[Dict] = None,
) -> Dict:
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
    params = default_params.copy()
    params.update(user_params)

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


