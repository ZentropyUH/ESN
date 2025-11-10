import inspect
import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin
from typing import Union as TypingUnion

import yaml
from keras.src.saving import deserialize_keras_object as keras_deserialize

from .config_models import LayerConfig, ReadoutConfig, ReservoirConfig


# ---------------------------------------------------------------------
# 1. Utilities
# ---------------------------------------------------------------------
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


def load_default_config(name: str) -> Dict[str, Any]:
    """Load a bundled default YAML config from io/defaults.

    The default configs are stored in the io/defaults directory.

    Parameters
    ----------
    name : str
        The name of the config to load.

    Returns
    -------
    Dict[str, Any]
        The loaded configuration.

    Raises
    ------
    FileNotFoundError
        If the default config file does not exist.
    """
    config_path = resources.files("keras_reservoir_computing.io.defaults").joinpath(f"{name}.yaml")
    if not config_path.exists():
        # Try JSON as fallback
        config_path = resources.files("keras_reservoir_computing.io.defaults").joinpath(f"{name}.json")
        if not config_path.exists():
            raise FileNotFoundError(
                f"Default config '{name}' not found in io/defaults directory. "
                f"Expected '{name}.yaml' or '{name}.json'."
            )
        with config_path.open("r") as f:
            return json.load(f)
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def load_config(source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load YAML/JSON file or dict into a Python dict.

    Parameters
    ----------
    source : Union[str, Dict[str, Any]]
        Path to a YAML/JSON file, raw YAML/JSON string, or a dictionary.

    Returns
    -------
    Dict[str, Any]
        The loaded configuration.

    Raises
    ------
    ValueError
        If the source cannot be parsed as YAML or JSON.
    FileNotFoundError
        If the file path does not exist.
    """
    if isinstance(source, dict):
        return source.copy()
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


# ---------------------------------------------------------------------
# 2. Unified loader
# ---------------------------------------------------------------------
def load_object(config_or_path: Union[str, Dict[str, Any], LayerConfig], strict: bool = True) -> Any:
    """Deserialize and validate a Keras object (layer, initializer, optimizer, ...).

    Parameters
    ----------
    config_or_path : Union[str, Dict[str, Any], LayerConfig]
        YAML/JSON file path, raw YAML/JSON string, a dict config, or a LayerConfig instance.
    strict : bool, optional
        Whether to enforce type validation (default: True).

    Returns
    -------
    Any
        The instantiated Keras object.

    Raises
    ------
    KeyError
        If the config is missing 'class_name'.
    ValueError
        If the object cannot be deserialized.
    TypeError
        If strict validation fails.
    """
    # Handle LayerConfig instances
    if isinstance(config_or_path, LayerConfig):
        config = config_or_path.to_dict()
    else:
        config = load_config(config_or_path)

    # Early sanity check for class_name
    if "class_name" not in config:
        raise KeyError("Invalid config: missing 'class_name'")

    try:
        obj = keras_deserialize(config)
    except Exception as e:
        raise ValueError(
            f"Failed to deserialize class '{config['class_name']}': {e}"
        ) from e

    _validate_config(obj.__class__, config.get("config", {}), strict=strict)
    return obj


# ---------------------------------------------------------------------
# 3. Configuration loaders with Pydantic validation
# ---------------------------------------------------------------------
def load_reservoir_config(
    source: Optional[Union[str, Dict[str, Any], ReservoirConfig]] = None
) -> ReservoirConfig:
    """Load and validate a reservoir configuration.

    Parameters
    ----------
    source : Union[str, Dict[str, Any], ReservoirConfig], optional
        Path to a YAML/JSON file, raw YAML/JSON string, a dict config,
        a ReservoirConfig instance, or None to load default. Default is None.

    Returns
    -------
    ReservoirConfig
        Validated reservoir configuration.

    Raises
    ------
    pydantic.ValidationError
        If the configuration is invalid.
    """
    if source is None:
        config_dict = load_default_config("reservoir")
    elif isinstance(source, ReservoirConfig):
        return source
    else:
        config_dict = load_config(source)

    return ReservoirConfig.from_dict(config_dict)


def load_readout_config(
    source: Optional[Union[str, Dict[str, Any], ReadoutConfig]] = None
) -> ReadoutConfig:
    """Load and validate a readout configuration.

    Parameters
    ----------
    source : Union[str, Dict[str, Any], ReadoutConfig], optional
        Path to a YAML/JSON file, raw YAML/JSON string, a dict config,
        a ReadoutConfig instance, or None to load default. Default is None.

    Returns
    -------
    ReadoutConfig
        Validated readout configuration.

    Raises
    ------
    pydantic.ValidationError
        If the configuration is invalid.
    """
    if source is None:
        config_dict = load_default_config("readout")
    elif isinstance(source, ReadoutConfig):
        return source
    else:
        config_dict = load_config(source)

    return ReadoutConfig.from_dict(config_dict)
