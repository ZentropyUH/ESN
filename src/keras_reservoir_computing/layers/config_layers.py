"""
Pydantic models and validation logic for every user‑supplied configuration.

Adding a new initializer later?  ➜  just register it with Keras (or with the
`krc>` prefix) and it will be accepted automatically – no code changes needed.
"""

import inspect
import json
from typing import Any, Dict, Union

import tensorflow as tf
from pydantic import BaseModel, Field, field_validator, model_validator

_CUSTOM_PREFIX = "krc>"  # prefix for your custom initializers


def _resolve_initializer(name: str, params: Dict[str, Any]) -> tf.keras.initializers.Initializer:
    """Return a concrete `tf.keras.initializers.Initializer` or raise."""
    try:
        return tf.keras.initializers.get({"class_name": name, "config": params})
    except ValueError:
        try:
            return tf.keras.initializers.get({"class_name": f"{_CUSTOM_PREFIX}{name}", "config": params})
        except ValueError as e:
            raise ValueError(f"Initializer '{name}' not found or parameters invalid - {e}")

def load_user_config(config: Union[str, Dict]) -> Dict:
    """
    Load JSON config from a file path or take an existing dict.
    """
    if isinstance(config, str):
        try:
            with open(config, "r") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading config from {config} - {e}") from e
    elif isinstance(config, dict):
        return config



class InitializerConfig(BaseModel):
    """Generic schema for *any* initializer, built-ins or custom."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    # -------- dynamic validation -------------------------------------------------
    @model_validator(mode='after')
    def _validate_and_warn(self):
        name, params = self.name, self.params

        # 1. Ensure the initializer exists (custom or builtin)
        _resolve_initializer(name, params)

        # 2. Make sure the user didn't supply garbage kwargs
        try:
            # grab the class object (works for custom too)
            inst = _resolve_initializer(name, {})  # instantiate with defaults
            cls_obj = inst.__class__
            sig = inspect.signature(cls_obj.__init__)
            allowed = {p.name for p in sig.parameters.values() if p.name != "self"}
            unknown = set(params) - allowed
            if unknown:
                raise ValueError(f"Unknown parameter(s) {unknown} for initializer '{name}'")
        except Exception as e:
            # propagate any meaningful error
            raise ValueError(str(e)) from e

        return self
    # -----------------------------------------------------------------------------


class ESNConfig(BaseModel):
    # ---- core reservoir parameters ---------------------------------------------
    units: int = 10
    feedback_dim: int = 1
    input_dim: int = 0
    leak_rate: float = Field(1.0, ge=0.0, le=1.0)
    activation: str = "tanh"
    dtype: str = "float32"

    # ---- initializers -----------------------------------------------------------
    input_initializer: InitializerConfig = Field(
        default_factory=lambda: InitializerConfig(name="zeros")
    )
    feedback_initializer: InitializerConfig = Field(
        default_factory=lambda: InitializerConfig(
            name="PseudoDiagonalInitializer",
            params={"sigma": 0.5, "binarize": False, "seed": None},
        )
    )
    feedback_bias_initializer: InitializerConfig = Field(
        default_factory=lambda: InitializerConfig(name="zeros")
    )
    kernel_initializer: InitializerConfig = Field(
        default_factory=lambda: InitializerConfig(
            name="WattsStrogatzGraphInitializer",
            params={
                "k": 6,
                "p": 0.2,
                "directed": True,
                "self_loops": True,
                "tries": 100,
                "spectral_radius": 0.9,
                "seed": None,
            },
        )
    )

    # ---- global validation ------------------------------------------------------
    @model_validator(mode='after')
    def _spectral_radius_vs_leak(self):
        leak = self.leak_rate
        ki = self.kernel_initializer
        sr = ki.params.get("spectral_radius")
        if sr is not None and sr < 1.0 - leak:
            raise ValueError("spectral_radius must be > 1 - leak_rate.")
        return self
    # -----------------------------------------------------------------------------


class ReadOutConfig(BaseModel):
    kind: str = "ridge"           # 'ridge' or 'mpenrose'
    units: int
    alpha: float | None = None    # only for ridge
    trainable: bool = True
    name: str | None = None
    dtype: str = "float32"

    # guard the 'kind'
    @field_validator("kind")
    def _kind_ok(cls, v):  # noqa: N805
        if v not in {"ridge", "mpenrose"}:
            raise ValueError("kind must be 'ridge' or 'mpenrose'")
        return v
