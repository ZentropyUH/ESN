"""
Factory helpers that turn user JSON/YAML → validated Pydantic models → real
Keras layers.  Every reservoir or readout built here is guaranteed valid.
"""

from typing import Dict, Optional, Union

import tensorflow as tf

from keras_reservoir_computing.layers import ESNReservoir
from keras_reservoir_computing.layers.config_layers import (
    ESNConfig,
    InitializerConfig,
    ReadOutConfig,
    load_user_config,
)
from keras_reservoir_computing.layers.readouts.base import ReadOut
from keras_reservoir_computing.layers.readouts.moorepenrose import (
    MoorePenroseReadout,
)
from keras_reservoir_computing.layers.readouts.ridge import RidgeSVDReadout


# ------------------------------------------------------------------------------
# helper – turn an InitializerConfig into a concrete tf initializer
# ------------------------------------------------------------------------------

def _build_initializer(cfg: InitializerConfig) -> tf.keras.initializers.Initializer:
    try:
        return tf.keras.initializers.get({"class_name": cfg.name, "config": cfg.params})
    except ValueError:
        try:
            return tf.keras.initializers.get(
                {"class_name": f"krc>{cfg.name}", "config": cfg.params}
            )
        except ValueError as e:
            raise ValueError(f"Initializer '{cfg.name}' not found - {e}") from e


# ------------------------------------------------------------------------------
# Reservoir
# ------------------------------------------------------------------------------

def ESNReservoir_builder(
    user_config: Union[str, Dict],
    overrides: Optional[Dict] = None,
) -> ESNReservoir:
    """
    Build an ESNReservoir layer from any JSON/dict config.

    Precedence: **overrides > user_config > built‑in defaults**
    """
    # 1. raw dict ---------------------------------------------------------------
    raw_cfg = load_user_config(user_config)

    # 2. overrides have highest priority
    if overrides:
        raw_cfg |= overrides                        # Python ≥3.9 dict merge

    # 3. validate & fill with defaults -----------------------------------------
    cfg = ESNConfig(**raw_cfg)

    # 4. adjust spectral radius on effective matrix ----------------------------
    ki = cfg.kernel_initializer
    sr = ki.params.get("spectral_radius")
    if sr is not None:
        W_sr = (sr - 1.0 + cfg.leak_rate) / cfg.leak_rate
        ki.params["spectral_radius"] = W_sr

    # 5. final dict with real tf objects ---------------------------------------
    final: Dict = cfg.model_dump()

    for k in (
        "input_initializer",
        "feedback_initializer",
        "feedback_bias_initializer",
        "kernel_initializer",
    ):
        final[k] = _build_initializer(InitializerConfig(**final[k]))  # type: ignore[arg-type]

    return ESNReservoir(**final)



def ReadOut_builder(
    user_config: Union[str, Dict],
    overrides: Optional[Dict] = None,
) -> ReadOut:
    """
    Build the requested read-out layer (ridge or Moore-Penrose).
    """
    raw_cfg = load_user_config(user_config)
    if overrides:
        raw_cfg |= overrides

    cfg = ReadOutConfig(**raw_cfg)
    cfg_dict = cfg.model_dump(exclude={"kind"})

    if cfg.kind == "moorepenrose":
        return MoorePenroseReadout(**cfg_dict)

    if cfg.kind == "ridge":
        return RidgeSVDReadout(**cfg_dict)

    raise ValueError(f"Invalid read-out kind: {cfg.kind}")


__all__ = ["ESNReservoir_builder", "ReadOut_builder"]


def __dir__() -> list[str]:  # noqa: D401
    return __all__
