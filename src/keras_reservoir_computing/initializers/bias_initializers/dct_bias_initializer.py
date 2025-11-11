import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="krc", name="DCTOneBiasInitializer")
class DCTOneBiasInitializer(tf.keras.initializers.Initializer):
    """
    Deterministic bias initializer for reservoirs based on the first DCT mode.

    This creates a smooth, topology-agnostic bias vector:

        b[j] = mu + cos(pi * (j + 0.5) / n)

    for j = 0..n-1.

    After construction the vector is L2-normalized to 'gain'.

    Why this works
    --------------
    This pattern pushes different units to different operating points,
    breaking symmetry and keeping tanh away from the unstable linear regime
    around zero (tanh'(0)=1). It behaves like a static form of intrinsic
    plasticity: stabilizes dynamics, increases activation diversity, and
    improves reservoir mixing. Unlike random biases, it is fully
    deterministic and reproducible, and does not depend on any specific
    reservoir topology (ring, dendrocycle, random, small-world, etc.).

    Parameters
    ----------
    mu : float
        Constant offset added before normalization. Small positive values
        (e.g. 0.05 - 0.2) push all units gently away from zero activation.
    gain : float
        Desired L2 norm of the final bias vector.

    Notes
    -----
    - Works for any reservoir structure.
    - No randomness is used.
    - Recommended defaults: mu=0.1, gain=1.0.
    """
    def __init__(
        self, 
        mu: float = 0.1, 
        gain: float = 1.0
    ):
        self.mu = float(mu)
        self.gain = float(gain)

    def __call__(self, shape, dtype=None):
        if not (isinstance(shape, (tuple, list)) and len(shape) == 1):
            raise ValueError(f"Expected shape (units,), got {shape}")

        n = int(shape[0])
        tf_dtype = tf.as_dtype(dtype or tf.float32)
        np_dtype = tf_dtype.as_numpy_dtype

        j = np.arange(n, dtype=np.float64)

        # first DCT mode (smoothest non-constant)
        b = self.mu + np.cos(np.pi * (j + 0.5) / n)

        # normalize
        norm = float(np.linalg.norm(b))
        if norm > 0:
            b = (self.gain / norm) * b

        return tf.convert_to_tensor(b.astype(np_dtype), dtype=tf_dtype)

    def get_config(self):
        return {"mu": self.mu, "gain": self.gain}