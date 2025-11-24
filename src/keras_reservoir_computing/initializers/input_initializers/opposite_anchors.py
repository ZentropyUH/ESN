import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="OppositeAnchorsInputInitializer")
class OppositeAnchorsInputInitializer(tf.keras.initializers.Initializer):
    """

    Initializer that connects each input channel with two opposite anchor nodes on the n-node ring.

    The weights are equal on both anchors (gain 'gain' normalized by sqrt(2), so total channel energy equals 'gain').
    If j0 == j1 (n = 1 or degenerate case), all the weight is used on that single node.

    Parameters
    ----------
    gain : float
        Global input gain. Defaults to  1.0.
    """

    def __init__(self, gain: float = 1.0) -> None:
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be a positive finite float.")
        self.gain = float(gain)

    def __call__(self, shape, dtype=None):
        if not (isinstance(shape, (tuple, list)) and len(shape) == 2):
            raise ValueError(f"shape must be (m, n); received: {shape!r}")
        m, n = int(shape[0]), int(shape[1])
        if m <= 0 or n <= 0:
            raise ValueError(f"m and n must be positive; received: (m={m}, n={n})")

        tf_dtype = tf.as_dtype(dtype or tf.float32)
        np_dtype = tf_dtype.as_numpy_dtype

        B = np.zeros((m, n), dtype=np_dtype)
        half = n // 2

        # n == 1 or degenerate
        if n == 1:
            B[:, 0] = np_dtype(self.gain)
            return tf.convert_to_tensor(B, dtype=tf_dtype)

        # evenly spaced anchors on the semicircle
        j0 = np.floor((np.arange(m) + 0.5) * half / m).astype(int)
        j1 = (j0 + half) % n

        w = np_dtype(self.gain / np.sqrt(2.0))
        B[np.arange(m), j0] = w
        B[np.arange(m), j1] = -w  # Consider putting here a negative sign.

        return tf.convert_to_tensor(B, dtype=tf_dtype)

    def get_config(self):
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        base = super().get_config()
        base.update(
            {
                "gain": self.gain,
            }
        )
        return base
