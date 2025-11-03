from typing import Optional

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="RandomInputInitializer")
class RandomInputInitializer(tf.keras.Initializer):
    """Simple random initializer.

    This initializer initializes the weights of the recurrent layer with random values
    from a uniform distribution between -1 and 1. Controls the spectral radius of the
    recurrent layer.

    Parameters
    ----------
    input_scale : float, optional
        If provided, the matrix will be scaled to have this input scale.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tf.Tensor
        A random matrix with the specified shape.

    Notes
    -----
    - The matrix is scaled to have the specified spectral radius if provided.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers.input_initializers import RandomInputInitializer
    >>> initializer = RandomInputInitializer(input_scale=1.0)
    >>> matrix = initializer((10, 10))
    >>> print(matrix)
    """
    def __init__(
        self,
        input_scaling: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the initializer with specified parameters."""
        self.input_scaling = input_scaling
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple, dtype=tf.float32) -> tf.Tensor:
        """Generate a random matrix with the specified shape.

        Parameters
        ----------
        shape : tuple
            Shape of the matrix to create.

        Returns
        -------
        tf.Tensor
            Random matrix with the specified shape.
        """
        dims = tf.TensorShape(shape).as_list()  # -> list[int|None]

        if dims is None:
            raise ValueError("Rank of shape unknown at initialization time.")
        if len(dims) == 1:
            rows, cols = int(dims[0]), 1
        elif len(dims) == 2:
            rows, cols = map(int, dims)
        else:
            raise ValueError(f"Shape must be 1D or 2D, got {shape}")

        # Generate random values
        W_in = self.rng.uniform(-1, 1, (rows, cols))

        # Translate the following to numpy
        if self.input_scaling is not None:
            W_in *= self.input_scaling

        # Convert to tf.Tensor
        W_in = tf.convert_to_tensor(W_in)

        # Cast to dtype
        W_in = tf.cast(W_in, dtype)

        return W_in

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
