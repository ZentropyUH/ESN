from typing import Optional
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="DendrocycleInputInitializer")
class DendrocycleInputInitializer(tf.keras.Initializer):
    """
    Keras initializer for input weight matrices of dendro-cycle reservoirs.

    Generates a matrix W_in of shape (N, M) or (M, N) depending on the layer’s
    expected shape, where only the core (cycle) nodes — the first C = round(c*N)
    rows — receive input connections.  All other entries are zero.

    Parameters
    ----------
    c : float or None
        Fraction of nodes forming the cycle (0 < c <= 1). Provide either c or C.
    C : int or None
        Number of cycle (core) nodes. If provided, c is ignored.
    input_scaling : float, optional
        Half-width of the uniform distribution U[-input_scaling, input_scaling]. Default 1.0.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    tf.Tensor
        Input weight matrix with non-zeros only in the core node columns.

    Examples
    --------
    >>> w_init = DendrocycleInputInitializer(c=0.2, input_scaling=0.5, seed=42)
    >>> W = w_init((100, 8))    # N=100, M=8
    >>> W.shape
    (100, 8)
    """

    def __init__(
        self,
        c: Optional[float] = None,
        C: Optional[int] = None,
        input_scaling: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if (c is None) == (C is None):
            raise ValueError("Provide exactly one of c or C.")
        self.c = c
        self.C = C
        self.input_scaling = input_scaling
        self.seed = seed
        super().__init__()

    # ----------------------------------------------------------------------
    def __call__(self, shape: tuple, dtype: Optional[tf.DType] = None) -> tf.Tensor:
        """
        Build the input weight matrix given the target shape.
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

        N, M = rows, cols
        C = self.C
        if C is None:
            if not (0 < self.c <= 1):
                raise ValueError("c must be in (0, 1].")
            C = max(1, int(round(self.c * N)))
        if not (1 <= C <= N):
            raise ValueError("C must be in [1, N].")

        rng = np.random.default_rng(self.seed)
        Win = np.zeros((N, M), dtype=np.float64)

        # Case 1: fewer inputs than cores
        if M <= C:
            mapping = [int(np.floor(i * M / C)) for i in range(C)]
            for core_idx, input_idx in enumerate(mapping):
                Win[core_idx, input_idx] = rng.uniform(-self.input_scaling, self.input_scaling)
        # Case 2: more inputs than cores
        else:
            mapping = [int(np.floor(i * C / M)) for i in range(M)]
            for input_idx, core_idx in enumerate(mapping):
                Win[core_idx, input_idx] = rng.uniform(-self.input_scaling, self.input_scaling)

        return tf.convert_to_tensor(Win, dtype=(dtype or tf.keras.backend.floatx()))

    # ----------------------------------------------------------------------
    def get_config(self) -> dict:
        """
        Return configuration dictionary for serialization.
        """
        base = super().get_config()
        base.update(
            {
                "c": self.c,
                "C": self.C,
                "input_scaling": self.input_scaling,
                "seed": self.seed,
            }
        )
        return base
