from typing import Optional

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="krc", name="DenseBinaryInitializer"
)
class DenseBinaryInitializer(tf.keras.Initializer):
    """
    An initializer that generates a binary matrix with values in {-1, 1}.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tf.Tensor
        The initialized weight matrix matching the requested shape.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import DenseBinaryInitializer
    >>> w_init = DenseBinaryInitializer(seed=42)
    >>> w = w_init((5, 10))
    >>> print(w)
    # A 5x10 matrix with values in {-1, 1}.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the initializer."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple, dtype=None) -> tf.Tensor:
        W = self.rng.choice([-1.0, 1.0], size=shape)
        W = tf.convert_to_tensor(W, dtype=dtype)
        return W

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        config = {"seed": self.seed}
        base_config = super().get_config()
        base_config.update(config)
        return base_config
