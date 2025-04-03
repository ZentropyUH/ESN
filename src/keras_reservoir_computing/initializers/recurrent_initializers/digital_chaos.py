from typing import Optional

import numpy as np
import tensorflow as tf

from keras_reservoir_computing.initializers.helpers import spectral_radius_hybrid


@tf.keras.utils.register_keras_serializable(package="krc", name="DigitalChaosInitializer")
class DigitalChaosInitializer(tf.keras.Initializer):
    r"""
    Initializes a sparse adjacency matrix based on a digital chaotic system.

    This initializer constructs a sparse adjacency matrix \( A \) of size \( 2^{2D} \times 2^{2D} \)
    using a digital chaotic system. The system evolves according to bitwise state transition rules
    influenced by random sequences \( s(n) \) and \( u(n) \).

    The recurrence equations defining the chaotic state evolution are:

    .. math::
        x_1(n) = [ x_1(n-1) \wedge s(n) ] \vee [ \overline{x_1(n-1)} \wedge \overline{x_2(n-1)} \wedge s(n) ]

    .. math::
        x_2(n) = [ x_2(n-1) \wedge u(n) ] \vee \left\{ \left( x_1(n-1) \vee (x_1(n-1) \wedge x_2(n-1)) \right)
        \wedge \overline{ \left( x_1(n-1) \vee (x_1(n-1) \wedge x_2(n-1)) \right) } \wedge u(n) \right\}

    where:

    - \( x_1, x_2 \) represent the system state variables (each a \( D \)-bit integer).
    - \( s(n), u(n) \) are independent \( D \)-bit random sequences drawn at each time step.
    - \( \wedge \) represents the bitwise AND operation.
    - \( \vee \) represents the bitwise OR operation.
    - \( \overline{x} \) represents the bitwise NOT operation (restricted to \( D \) bits).

    **Matrix Structure and Sparsity Considerations**

    The constructed adjacency matrix has a size of \( M = 2^{2D} \), meaning its full storage
    becomes infeasible for large \( D \) due to exponential growth. To mitigate this, the matrix
    is stored in a **sparse representation** using `scipy.sparse.lil_matrix`, ensuring efficient memory
    usage.

    The number of outgoing connections per state can be controlled via:

    - `samples_per_state`: Fixes the number of transitions \( K \) per state.
    - `non_zero_percentage`: Sets \( K \) dynamically as a fraction of \( M \).

    **Caveats:**
    - The adjacency matrix is **strictly constrained** to size \( 2^{2D} \times 2^{2D} \), meaning
      small \( D \) leads to limited expressivity, while large \( D \) is computationally expensive.
    - Using a **dense representation** for \( A \) is **not feasible** for \( D \geq 10 \)
      due to excessive memory requirements (e.g., \( D=10 \) leads to a \( 1\text{ TiB} \) matrix).
    - The ESN reservoir **does not require storing the full matrix**—only the dynamically
      sampled state transitions matter for practical usage.

    Parameters
    ----------
    samples_per_state : int, optional
        Number of random transitions \( K \) per state. Defaults to 5.
    non_zero_percentage : float, optional
        Fraction of \( M \) to determine \( K \). If provided, `samples_per_state` is ignored.
    spectral_radius : float, optional
        Spectral radius of the reservoir matrix. If None, the matrix is not scaled
        to have a specific spectral radius.

    Returns
    -------
    tf.Tensor
        The generated adjacency matrix converted to a dense TensorFlow tensor.

    Raises
    ------
    ValueError
        If the provided shape does not match the expected \( (M, M) \).

    Notes
    -----
    - This method follows a **controlled-random connectivity** approach, ensuring strong
      connectivity while keeping the ESN reservoir structured.
    - If `non_zero_percentage` is not specified, the number of transitions per state remains fixed.
    - The use of a **bitwise chaotic map** instead of a standard random ESN reservoir matrix
      ensures a structured but unpredictable recurrence topology.

    References
    ----------
    .. M. Xie, Q. Wang, and S. Yu, “Time Series Prediction of ESN Based on Chebyshev Mapping and Strongly Connected Topology,” Neural Process Lett, vol. 56, no. 1, p. 30, Feb. 2024, doi: 10.1007/s11063-024-11474-7.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import DigitalChaosInitializer
    >>> w_init = DigitalChaosInitializer(samples_per_state=5, non_zero_percentage=0.1, spectral_radius=0.9)
    >>> w = w_init((16, 16))
    >>> print(w)
    # A 16x16 adjacency matrix initialized using digital chaos.

    """

    def __init__(
        self,
        D: int = 2,
        samples_per_state: int = 5,
        non_zero_percentage: float = None,
        spectral_radius: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.D = D
        self.samples_per_state = samples_per_state
        self.non_zero_percentage = non_zero_percentage
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple, dtype=None) -> tf.Tensor:
        if self.D is None:
            raise ValueError("D must be provided when calling the initializer.")

        M = 2 ** (2 * self.D)

        if shape != (M, M):
            raise ValueError(f"Shape mismatch: expected ({M}, {M}), but got {shape}")

        def bitwise_not_dbits(x, D):
            mask = (1 << D) - 1
            return x ^ mask

        def next_state_2d_digital(x1, x2, s, u, D):
            x1_bar = bitwise_not_dbits(x1, D)
            x2_bar = bitwise_not_dbits(x2, D)

            x1_next = (x1 & s) | (x1_bar & x2_bar & s)
            a = x1 | (x1 & x2)
            a_bar = bitwise_not_dbits(a, D)
            x2_next = (x2 & u) | (a & a_bar & u)

            return x1_next & ((1 << D) - 1), x2_next & ((1 << D) - 1)

        W_recurrent = np.zeros((M, M), dtype=np.float32)

        if self.non_zero_percentage is not None:
            K = int(M * self.non_zero_percentage)
        else:
            K = self.samples_per_state

        for old_state in range(M):
            x1_old = old_state >> self.D
            x2_old = old_state & ((1 << self.D) - 1)

            for _ in range(K):
                s = self.rng.integers(0, 1 << self.D)
                u = self.rng.integers(0, 1 << self.D)
                x1_new, x2_new = next_state_2d_digital(x1_old, x2_old, s, u, self.D)
                new_state = (x1_new << self.D) | x2_new
                W_recurrent[old_state, new_state] = self.rng.choice([-1, 1])

        W_recurrent = tf.convert_to_tensor(W_recurrent, dtype=dtype)

        if self.spectral_radius is not None:
            sr = spectral_radius_hybrid(W_recurrent)
            W_recurrent = W_recurrent * self.spectral_radius / sr

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
            "D": self.D,
            "samples_per_state": self.samples_per_state,
            "non_zero_percentage": self.non_zero_percentage,
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config
