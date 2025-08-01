import inspect
import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Union, get_args, get_origin
from typing import Union as TypingUnion

import yaml
from keras.src.saving import deserialize_keras_object as keras_deserialize


# ---------------------------------------------------------------------
# 1. Utilities
# ---------------------------------------------------------------------
def _load_config(source: Union[str, Dict]) -> Dict:
    """Load YAML/JSON file or dict into a Python dict.

    Parameters
    ----------
    source : Union[str, Dict]
        Path to a YAML/JSON file or a dictionary.

    Returns
    -------
    Dict
        The loaded configuration.
    """
    if isinstance(source, dict):
        return source
    path = Path(source)
    if isinstance(source, str) and path.is_file():
        text = path.read_text()
    else:
        text = source
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as yaml_err:
        try:
            return json.loads(text)
        except json.JSONDecodeError as json_err:
            raise ValueError(
                f"Failed to parse config as YAML (error: {yaml_err}) and as JSON (error: {json_err})."
            ) from json_err


def _is_instance_of(value, annotation) -> bool:
    """Check if a value is an instance of a given annotation.

    Parameters
    ----------
    value : Any
        The value to check.
    annotation : Any
        The annotation to check against.

    Returns
    -------
    bool
        True if the value is an instance of the annotation, False otherwise.
    """
    origin = get_origin(annotation)
    if origin is None:
        return isinstance(value, annotation)
    if origin is list:
        return isinstance(value, list) and all(
            _is_instance_of(v, get_args(annotation)[0]) for v in value
        )
    if origin is TypingUnion:
        return any(_is_instance_of(value, arg) for arg in get_args(annotation))
    return True


def _validate_config(cls, cfg: Dict, strict: bool = True) -> None:
    """Validate a configuration dictionary against a class's signature.

    Parameters
    ----------
    cls : Any
        The class to validate against.
    cfg : Dict
        The configuration dictionary to validate.
    strict : bool, optional
        If True (default), enforces type checks strictly.
        If False, only checks required arguments exist.

    Raises
    ------
    ValueError
        If a required argument is missing.
    TypeError
        If strict=True and a type mismatch is found.
    """
    sig = inspect.signature(cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if name not in cfg and param.default is param.empty:
            raise ValueError(f"{cls.__name__} is missing required arg: {name}")
        if strict and name in cfg and param.annotation is not param.empty:
            expected = param.annotation
            if not _is_instance_of(cfg[name], expected):
                raise TypeError(
                    f"{cls.__name__}.{name} expects {expected}, got {type(cfg[name])}"
                )


def load_default_config(name: str) -> dict:
    """Load a bundled default YAML config from io/defaults.

    The default configs are stored in the io/defaults directory.

    Parameters
    ----------
    name : str
        The name of the config to load.

    Returns
    -------
    Dict
        The loaded configuration.
    """
    with resources.files("keras_reservoir_computing.io.defaults").joinpath(f"{name}.yaml").open("r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------
# 2. Unified loader
# ---------------------------------------------------------------------
def load_object(config_or_path: Union[str, Dict], strict: bool = True) -> Any:
    """Deserialize and validate a Keras object (layer, initializer, optimizer, ...).

    Parameters
    ----------
    config_or_path : Union[str, Dict]
        YAML/JSON file path, raw YAML/JSON string, or a dict config.
    strict : bool, optional
        Whether to enforce type validation (default: True).

    Returns
    -------
    Any
        The instantiated Keras object.
    """
    config = _load_config(config_or_path)

    # Early sanity check for class_name
    if "class_name" not in config:
        raise KeyError("Invalid config: missing 'class_name'")

    try:
        obj = keras_deserialize(config)
    except Exception as e:
        raise ValueError(f"Failed to deserialize class '{config['class_name']}': {e}") from e

    _validate_config(obj.__class__, config.get("config", {}), strict=strict)
    return obj
