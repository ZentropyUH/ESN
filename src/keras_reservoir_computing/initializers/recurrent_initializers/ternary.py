from typing import Optional

import keras
import numpy as np
import tensorflow as tf
from keras import Initializer
from keras_reservoir_computing.initializers.helpers import spectral_radius_hybrid


@keras.saving.register_keras_serializable(package="krc", name="TernaInitializer")
class TernaryInitializer(Initializer):
    """
    Ternary Initializer for creating weight matrices with values in {-1, 0, 1}.

    This initializer creates a recurrent weight matrix with ternary values (-1, 0, 1)
    according to the specified probabilities. It can optionally scale the resulting
    matrix to have a specific spectral radius.

    Parameters
    ----------
    zero_p : float, default=0.5
        The probability of a weight being 0.
    neg_p : float, default=0.25
        The probability of a weight being -1.
    pos_p : float, default=0.25
        The probability of a weight being 1.
    spectral_radius : float, optional
        If provided, the weights will be scaled to achieve this spectral radius.
    seed : int, optional
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If the sum of the probabilities (zero_p + neg_p + pos_p) does not equal 1.

    Notes
    -----
    The ternary weight distribution can be useful for creating sparse, efficient
    reservoir networks with discrete weight values.

    References
    ----------

    H. Jaeger, “The 'echo state' approach to analysing and training recurrent neural networks - with an Erratum note”.


    Examples
    --------
    >>> from keras_reservoir_computing.initializers import TernaryInitializer
    >>> import tensorflow as tf
    >>> import numpy as np
    >>>
    >>> # Create a ternary initializer with custom probabilities
    >>> initializer = TernaryInitializer(zero_p=0.7, neg_p=0.15, pos_p=0.15, spectral_radius=0.9, seed=42)
    >>>
    >>> # Use it to initialize a recurrent kernel
    >>> weights = initializer((100, 100), dtype=tf.float32)
    >>>
    >>> # Verify distribution of values
    >>> unique, counts = np.unique(weights.numpy(), return_counts=True)
    >>> print(dict(zip(unique, counts)))  # Should approximate the specified probabilities

    """

    def __init__(
        self,
        zero_p: float = 0.95,
        neg_p: float = 0.025,
        pos_p: float = 0.025,
        spectral_radius: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.zero_p = zero_p
        self.neg_p = neg_p
        self.pos_p = pos_p
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if zero_p + neg_p + pos_p != 1:
            raise ValueError("The sum of the probabilities must be 1")

    def __call__(self, shape, dtype=None):
        W_recurrent = self.rng.choice(
            [-1, 0, 1], size=shape, p=[self.neg_p, self.zero_p, self.pos_p]
        )
        W_recurrent = tf.convert_to_tensor(W_recurrent, dtype=dtype)

        if self.spectral_radius is not None:
            sr = spectral_radius_hybrid(W_recurrent)
            W_recurrent = W_recurrent * (self.spectral_radius / sr)

        return W_recurrent

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        config = {
            "zero_p": self.zero_p,
            "neg_p": self.neg_p,
            "pos_p": self.pos_p,
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config
