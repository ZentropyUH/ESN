import tensorflow as tf
from typing import Union, Sequence
import numpy as np


@tf.keras.utils.register_keras_serializable(package="krc", name="ChainOfNeuronsInputInitializer")
class ChainOfNeuronsInputInitializer(tf.keras.initializers.Initializer):
    """
    Input initializer for ChainOfNeurons reservoirs.

    - 'features' = number of chains (must equal input_dim at call).
    - 'weights' is either:
        * a single float: same weight for all input→chain pairs
        * a list/tuple/array of floats of length 'features':
          one deterministic weight per input/chain.

    Shape expected at call: (input_dim, units)
    - input_dim must equal 'features'
    - units must be a multiple of 'features'
    - each input i connects to the first unit of chain i with weight weights[i].
    """

    def __init__(
        self,
        features: int,
        weights: Union[float, Sequence[float]] = 1.0,
    ):
        if features < 1:
            raise ValueError(f"'features' must be >= 1, got {features}.")
        self.features = int(features)
        self.weights = weights

    def __call__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError(
                "ChainOfNeuronsInputInitializer expects a 2D shape "
                f"(input_dim, units), got {shape}."
            )

        input_dim = int(shape[0])
        units = int(shape[1])

        if input_dim != self.features:
            raise ValueError(
                f"input_dim ({input_dim}) must equal 'features' ({self.features}) "
                "to have one input per chain."
            )
        if units % self.features != 0:
            raise ValueError(
                f"Number of units ({units}) must be a multiple of 'features' "
                f"({self.features}) to align chains with inputs."
            )

        tf_dtype, np_dtype = self._resolve_dtype(dtype)
        W_in = np.zeros((input_dim, units), dtype=np_dtype)

        # Resolve per-input weights.
        if isinstance(self.weights, (list, tuple, np.ndarray)):
            if len(self.weights) != input_dim:
                raise ValueError(
                    "When 'weights' is a sequence, its length must equal input_dim; "
                    f"got len(weights)={len(self.weights)}, input_dim={input_dim}."
                )
            in_weights = [np_dtype(float(w)) for w in self.weights]
        else:
            w = np_dtype(float(self.weights))
            in_weights = [w] * input_dim

        block_len = units // self.features

        # Deterministic: input i → first unit of chain i.
        for i in range(input_dim):
            start = i * block_len
            W_in[i, start] = in_weights[i]

        return tf.convert_to_tensor(W_in, dtype=tf_dtype)

    @staticmethod
    def _resolve_dtype(dtype):
        """Return a tf.DType and matching NumPy dtype from a Keras initializer dtype argument."""
        if dtype is None:
            dtype = tf.keras.backend.floatx()
        tf_dtype = tf.as_dtype(dtype)
        np_dtype = tf_dtype.as_numpy_dtype
        return tf_dtype, np_dtype

    def get_config(self):
        return {
            "features": int(self.features),
            "weights": self.weights,
        }
