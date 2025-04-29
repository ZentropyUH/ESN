from typing import Optional

import numpy as np
import tensorflow as tf

from keras_reservoir_computing.initializers.helpers import spectral_radius_hybrid


@tf.keras.utils.register_keras_serializable(package="krc", name="RandomRecurrentInitializer")
class RandomRecurrentInitializer(tf.keras.Initializer):
    """Simple random initializer.

    This initializer initializes the weights of the recurrent layer with random values
    from a uniform distribution between -1 and 1. Controls the spectral radius of the
    recurrent layer.

    Parameters
    ----------
    density : float, default=0.3
        Float in [0,1] representing the target proportion of non-zero entries.
    spectral_radius : float, optional
        If provided, the matrix will be scaled to have this spectral radius.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tf.Tensor
        A random matrix with the specified shape.

    Notes
    -----
    - The matrix is scaled to have the specified spectral radius if provided.
    - The matrix is masked to control the density of non-zero entries.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers.recurrent_initializers import RandomRecurrentInitializer
    >>> initializer = RandomRecurrentInitializer(density=0.3, spectral_radius=1.0)
    >>> matrix = initializer((10, 10))
    >>> print(matrix)
    """
    def __init__(
        self,
        density: float = 0.3,
        spectral_radius: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the initializer with specified parameters."""
        self.density = density
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple, dtype=tf.float32) -> tf.Tensor:
        """Generate a random matrix with the specified shape. Controls the density of
        non-zero values in the matrix according to the density parameter applying a
        mask to the random values.

        Parameters
        ----------
        shape : tuple
            Shape of the matrix to create.

        Returns
        -------
        tf.Tensor
            Random matrix with the specified shape.
        """
        if len(shape) == 2 and shape[0] != shape[1]:
            raise ValueError("RandomRecurrentInitializer only supports square matrices.")
        # Generate random values
        values = self.rng.uniform(-1, 1, shape)

        # Generate mask to control density
        num_nonzeros = int(np.round(self.density * np.prod(shape)))
        indices = self.rng.choice(np.prod(shape), size=num_nonzeros, replace=False)
        mask = np.zeros(np.prod(shape), dtype=bool)
        mask[indices] = True
        mask = mask.reshape(shape)

        # Apply mask to control density
        W_r = values * mask

        # Scale to spectral radius if provided
        if self.spectral_radius is not None:
            try:
                sr = spectral_radius_hybrid(W_r)
                if sr > 0:  # Avoid division by zero
                    W_r = W_r * (self.spectral_radius / sr)
                else:
                    np.testing.assert_greater(
                        sr, 0.0,
                        err_msg="Spectral radius calculation returned zero or negative value."
                    )
            except Exception as e:
                print(f"Warning: Spectral radius calculation failed. Using matrix without scaling. Error: {e}")


        # Convert to tf.Tensor
        W_r = tf.convert_to_tensor(W_r)

        # Cast to dtype
        W_r = tf.cast(W_r, dtype)

        return W_r

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        config = {
            "density": self.density,
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config
