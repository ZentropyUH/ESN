import tensorflow as tf
from typing import Union, Sequence
import numpy as np

@tf.keras.utils.register_keras_serializable(package="krc", name="ChainOfNeuronsInitializer")
class ChainOfNeuronsInitializer(tf.keras.initializers.Initializer):
    """
    Reservoir initializer: block-diagonal collection of chains of neurons.

    - 'features' is the total number of chains.
    - 'weights' is either:
        * a single float: same weight for all chains
        * a list/tuple/array of floats of length 'features': one weight per chain

    Each chain is a simple delay line:
        x_{t+1}^{(k,i+1)} = w_k * x_t^{(k,i)}
    implemented as subdiagonal blocks with constant chain weight.
    """

    def __init__(
        self,
        features: int,
        weights: Union[float, Sequence[float]] = 0.9,
    ):
        if features < 1:
            raise ValueError(f"'features' must be >= 1, got {features}.")
        self.features = int(features)
        self.weights = weights

    def __call__(self, shape, dtype=None):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                "ChainOfNeuronsReservoirInitializer expects a square 2D shape "
                f"(units, units), got {shape}."
            )
        units = int(shape[0])
        if units % self.features != 0:
            raise ValueError(
                f"Number of units ({units}) must be a multiple of 'features' "
                f"({self.features}) to allocate equal-length chains."
            )

        tf_dtype, np_dtype = self._resolve_dtype(dtype)
        W = np.zeros((units, units), dtype=np_dtype)

        # Resolve per-chain weights.
        if isinstance(self.weights, (list, tuple, np.ndarray)):
            if len(self.weights) != self.features:
                raise ValueError(
                    f"Length of 'weights' ({len(self.weights)}) must equal 'features' "
                    f"({self.features}) when a sequence is provided."
                )
            chain_weights = [np_dtype(float(w)) for w in self.weights]
        else:
            w = np_dtype(float(self.weights))
            chain_weights = [w] * self.features

        block_len = units // self.features

        for k in range(self.features):
            lam = chain_weights[k]
            start = k * block_len
            end = start + block_len
            # Subdiagonal inside this block.
            for i in range(start, end - 1):
                W[i + 1, i] = lam

        return tf.convert_to_tensor(W, dtype=tf_dtype)

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
