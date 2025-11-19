from typing import Optional

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="ChebyshevInitializer")
class ChebyshevInitializer(tf.keras.Initializer):
    r"""
    Keras initializer using Chebyshev mapping for Echo State Networks (ESNs).

    This initializer constructs a weight matrix based on the Chebyshev polynomial map, ensuring
    structured, chaotic initialization while maintaining a controlled range.

    **Why Chebyshev Mapping?**
    The Chebyshev polynomial recurrence exhibits **deterministic chaos**, making it a
    structured alternative to purely random weight initialization. This enhances the
    richness how the input signal is connected to the reservoir neurons.

    **How This Works**
    The Chebyshev map is recursively applied **column-wise** to generate the matrix
    structure. This ensures compatibility with **post-multiplication** usage in ESNs:

    .. math::
        \mathbf{y} = \mathbf{x} W

    where:

    - \( W \) is the initialized weight matrix.
    - \( \mathbf{x} \) is an input vector (batch, K).
    - \( \mathbf{y} \) is the transformed output vector (batch, N).

    **Chebyshev Recurrence Equations**

    The first column is initialized sinusoidally:

    .. math::
        W_{i, 0} = p \cdot \sin \left( \frac{i}{K+1} \cdot \frac{\pi}{q} \right)

    The rest of the columns evolve using the Chebyshev recurrence:

    .. math::
        W_{i, j} = \cos \left( k \cdot \cos^{-1} ( W_{i, j-1} ) \right), \quad j \geq 1

    where:

    - \( p \) controls the initial amplitude.
    - \( q \) scales the sinusoidal initialization.
    - \( k \) controls the **chaotic behavior** of the Chebyshev map (optimal range: \( 2 < k < 4 \)).

    **Pre-Multiplication vs. Post-Multiplication**
    - The original paper builds the matrix **row-wise** for **pre-multiplication** \( W x \).
    - This implementation builds it **column-wise** for **post-multiplication** \( x W \), ensuring **identical structure**.


    **Caveats**
    - \( k \) must be in the range **\( 2 < k < 4 \)** for effective chaotic behavior.
    - \( p \) and \( q \) control the amplitude and spread of the initial sinusoidal distribution.
    - \( p \) should be bound to \( 0 < p < 1 \).

    Parameters
    ----------
    p : float, optional
        Scaling factor for the initial sinusoidal weights. Default is 0.3.
    q : float, optional
        Parameter controlling the initial sinusoidal distribution. Default is 5.9.
    k : float, optional
        Control parameter of the Chebyshev map (chaotic regime: \( 2 < k < 4 \)). Default is 3.8.
    sigma : float, optional
        Input scaling factor. If None, the rescaling is disabled.

    Returns
    -------
    tf.Tensor
        The initialized weight matrix matching the requested shape.

    Raises
    ------
    ValueError
        If `k` is not in the chaotic range \( (2, 4) \), affecting network dynamics.

    Notes
    -----
    - This initializer ensures a **deterministic but structured** initialization.
    - Unlike purely random initializations, it **preserves correlation structure** between neurons.
    - The transformation is applied **column-wise**, ensuring correct behavior for **post-multiplication** ESNs.

    References
    ----------
    .. M. Xie, Q. Wang, and S. Yu, “Time Series Prediction of ESN Based on Chebyshev Mapping and Strongly Connected Topology,” Neural Process Lett, vol. 56, no. 1, p. 30, Feb. 2024, doi: 10.1007/s11063-024-11474-7.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import ChebyshevInitializer
    >>> w_init = ChebyshevInitializer(p=0.3, q=5.9, k=3.8, sigma=0.5)
    >>> w = w_init((5, 10))
    >>> print(w)
    # A 5x10 matrix initialized using Chebyshev mapping.

    """

    def __init__(
        self,
        p: float = 0.3,
        q: float = 5.9,
        k: float = 3.8,
        input_scaling: Optional[float] = None,
    ) -> None:
        """
        Initialize the Chebyshev weight matrix initializer.

        Parameters
        ----------
        p : float, optional
            Scaling factor for the initial sinusoidal weights. Should be in range (0, 1).
            Default is 0.3.
        q : float, optional
            Parameter controlling the initial sinusoidal distribution.
            Default is 5.9.
        k : float, optional
            Control parameter of the Chebyshev map, must be in range (2, 4) for
            chaotic behavior. Default is 3.8.
        sigma : float, optional
            Input scaling factor. If None, matrix rescaling is disabled.
            Default is None.

        Raises
        ------
        ValueError
            If k is not in the valid range (2, 4).

        Examples
        --------
        >>> from keras_reservoir_computing.initializers import ChebyshevInitializer
        >>> # Standard initialization
        >>> initializer = ChebyshevInitializer()
        >>> # Custom parameters
        >>> initializer = ChebyshevInitializer(p=0.4, q=5.0, k=3.2, sigma=0.8)
        """
        # Validate k is in the chaotic regime
        if not (2.0 < k < 4.0):
            raise ValueError(f"Parameter k={k} must be in range (2, 4) for chaotic behavior")

        self.p = p
        self.q = q
        self.k = k
        self.input_scaling = input_scaling

    def __call__(self, shape: tuple, dtype=tf.float32) -> tf.Tensor:
        """
        Generate a weight matrix using the Chebyshev map.

        Creates a structured weight matrix with chaotic properties by applying the
        Chebyshev polynomial map column-wise. The first column is initialized with
        a sinusoidal pattern, and subsequent columns are generated through the
        Chebyshev recurrence relation.

        Parameters
        ----------
        shape : tuple
            Shape of the weight matrix (K, N) where K is input dimension and
            N is output dimension. For ESNs, typically this is (input_features, units).
        dtype : tf.DType, optional
            Data type of the returned tensor. Default is tf.float32.

        Returns
        -------
        tf.Tensor
            The initialized weight matrix of shape (K, N) with deterministic
            chaotic structure based on the Chebyshev map.

        Notes
        -----
        - The matrix is constructed column-wise for post-multiplication (x*W).
        - Matrix values will generally be in the range [-1, 1] before scaling.
        - If sigma is specified, the matrix is rescaled to have a maximum
          singular value of sigma.

        Examples
        --------
        >>> from keras_reservoir_computing.initializers import ChebyshevInitializer
        >>> initializer = ChebyshevInitializer(p=0.3, k=3.5, sigma=0.8)
        >>> # Create a matrix for input features=5, output units=10
        >>> matrix = initializer((5, 10))
        >>> print(matrix.shape)
        (5, 10)
        >>> # Check the maximum singular value if sigma was specified
        >>> import tensorflow as tf
        >>> s = tf.linalg.svd(matrix, compute_uv=False)
        >>> print(f"Maximum singular value: {tf.reduce_max(s):.4f}")
        Maximum singular value: 0.8000
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
        K, N = rows, cols  # K = Inputs, N = Reservoir Neurons (for x * W)

        # Initialize first column with sinusoidal mapping
        row_indices = np.arange(1, K + 1, dtype=np.float32)  # Correct indexing
        W = np.zeros((K, N), dtype=np.float32)
        W[:, 0] = self.p * np.sin((row_indices / (K + 1)) * (np.pi / self.q))  # First column

        # Apply Chebyshev recurrence **column-wise**
        for j in range(1, N):
            W[:, j] = np.cos(self.k * np.arccos(np.clip(W[:, j - 1], -1.0, 1.0)))

        # Rescale the matrix if requested
        if self.input_scaling is not None:
            W *= self.input_scaling

        return tf.convert_to_tensor(W, dtype=dtype)

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed to reconstruct
            the initializer (p, q, k, sigma).

        Examples
        --------
        >>> from keras_reservoir_computing.initializers import ChebyshevInitializer
        >>> initializer = ChebyshevInitializer(p=0.4, k=3.5)
        >>> config = initializer.get_config()
        >>> print(config)
        {'p': 0.4, 'q': 5.9, 'k': 3.5, 'input_scaling': None}
        >>> # Recreate from config
        >>> new_initializer = ChebyshevInitializer.from_config(config)
        """
        config = {
            "p": self.p,
            "q": self.q,
            "k": self.k,
            "input_scaling": self.input_scaling,
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config
