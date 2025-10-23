from typing import Optional

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="krc", name="RandomBinaryInitializer"
)
class RandomBinaryInitializer(tf.keras.Initializer):
    """
    An initializer that generates a binary matrix with values in {-1, 1}.

    Parameters
    ----------
    input_scaling : float, optional
        Scaling factor for the input matrix.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tf.Tensor
        The initialized weight matrix matching the requested shape.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import RandomBinaryInitializer
    >>> w_init = RandomBinaryInitializer(seed=42)
    >>> w = w_init((5, 10))
    >>> print(w)
    # A 5x10 matrix with values in {-1, 1}.
    """

    def __init__(
        self,
        input_scaling: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the initializer."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.input_scaling = input_scaling
    def __call__(self, shape: tuple, dtype=None) -> tf.Tensor:
        dims = tf.TensorShape(shape).as_list()  # -> list[int|None]

        if dims is None:
            raise ValueError("Rank of shape unknown at initialization time.")
        if len(dims) == 1:
            rows, cols = int(dims[0]), 1
        elif len(dims) == 2:
            rows, cols = map(int, dims)
        else:
            raise ValueError(f"Shape must be 1D or 2D, got {shape}")
        W = self.rng.choice([-1.0, 1.0], size=(rows, cols))
        W = tf.convert_to_tensor(W, dtype=dtype)
        if self.input_scaling is not None:
            W *= self.input_scaling
        return W

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        config = {
            "input_scaling": self.input_scaling,
            "seed": self.seed,
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config
