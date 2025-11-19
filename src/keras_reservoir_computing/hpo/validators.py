# ------------------------------------------------------------------
# Validation utilities
# ------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Any, List, Mapping
from keras_reservoir_computing.layers.readouts.base import ReadOut

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tensorflow as tf


def _validate_data_dict(data: Mapping[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys are present in the data dictionary.

    Parameters
    ----------
    data : Mapping[str, Any]
        The data dictionary to validate.
    required_keys : List[str]
        List of required keys that must be present.

    Raises
    ------
    KeyError
        If any required keys are missing.
    """
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"Missing required keys in data dictionary: {', '.join(missing)}. "
            f"Required keys: {', '.join(required_keys)}"
        )


def _validate_tensor_shapes(data: Mapping[str, Any]) -> None:
    """Validate that data tensors have consistent shapes.

    Parameters
    ----------
    data : Mapping[str, Any]
        The data dictionary containing tensors.

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent.
    """
    batch_keys = [
        "transient",
        "train",
        "train_target",
        "ftransient",
        "val",
        "val_target",
    ]
    batch_sizes: dict[str, int] = {}
    for key in batch_keys:
        if key in data:
            tensor = data[key]
            if hasattr(tensor, "shape") and len(tensor.shape) > 0:
                batch_sizes[key] = int(tensor.shape[0])
    if len(set(batch_sizes.values())) > 1:
        logger.warning(
            f"Inconsistent batch sizes detected: {batch_sizes}. "
            "This may cause issues during training."
        )


def _infer_readout_targets(
    model: "tf.keras.Model",
    train_target: "tf.Tensor",
) -> Mapping[str, "tf.Tensor"]:
    """Infer readout targets from model structure.

    Automatically detects ReadOut layers in the model and assigns
    training targets to them. For single ReadOut models, uses the
    provided train_target. For multiple ReadOuts, raises an error
    requesting explicit specification.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing ReadOut layers.
    train_target : tf.Tensor
        The training target tensor.

    Returns
    -------
    Mapping[str, tf.Tensor]
        Dictionary mapping ReadOut layer names to target tensors.

    Raises
    ------
    ValueError
        If no ReadOut layers are found or if multiple ReadOuts exist
        without explicit target specification.
    """
    readouts = [layer for layer in model.layers if isinstance(layer, ReadOut)]
    if len(readouts) == 0:
        raise ValueError(
            "No ReadOut layers found in the model. "
            "Ensure your model contains at least one ReadOut layer."
        )
    if len(readouts) == 1:
        return {readouts[0].name: train_target}
    raise ValueError(
        f"Multiple ReadOut layers found: {[layer.name for layer in readouts]}. "
        "Provide a 'readout_targets' mapping in the data dictionary."
    )
