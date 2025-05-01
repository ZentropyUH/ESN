from math import isqrt
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from keras_reservoir_computing.initializers.helpers import spectral_radius_hybrid

from .base import GraphInitializerBase


@tf.keras.utils.register_keras_serializable(
    package="krc", name="SpectralCascadeGraphInitializer"
)
class SpectralCascadeGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices composed of multiple disconnected cliques
    with decreasing spectral radius per clique.

    Each clique has size 1 to N (with total nodes n = N(N+1)/2), and is fully connected
    with deterministic weights in {+1, -1}, scaled so that:
        - Clique of size 1 has spectral radius 0
        - Clique of size 2 has spectral radius = `spectral_radius`
        - Larger cliques decay linearly in spectral radius down to sr / N

    Parameters
    ----------
    spectral_radius : float
        Target spectral radius for the 2-node clique (others are scaled accordingly).
    self_loops : bool, optional
        Whether to include self-loops of weight (-1)^i. Default: False.
    seed : int or None, optional
        Ignored. Present for compatibility.
    """
    def __init__(
        self,
        spectral_radius: float,
        self_loops: bool = False,
        seed: int = None,
    ) -> None:
        self.self_loops = self_loops
        super().__init__(spectral_radius=None, seed=seed)
        self.spectral_radius = spectral_radius

    def __call__(
        self,
        shape: Union[int, Tuple[int, int], List[int]],
        dtype: Optional[tf.dtypes.DType] = None,
    ) -> tf.Tensor:
        if isinstance(shape, int):
            shape = (shape, shape)
        elif (not isinstance(shape, (tuple, list)) or len(shape) != 2) or shape[0] != shape[1]:
            raise ValueError("The shape of the adjacency matrix should be square 2D.")

        n = shape[0]
        adj = self._generate_adjacency_matrix(n)

        # apply scaled per-clique spectral control
        return tf.convert_to_tensor(adj, dtype=dtype)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:

        D = 1 + 8 * n
        sqrt_D = isqrt(D)
        if sqrt_D * sqrt_D != D or (sqrt_D - 1) % 2 != 0:
            raise ValueError(f"{n} is not a triangular number (N(N+1)/2).")

        N = (sqrt_D - 1) // 2
        A = np.zeros((n, n), dtype=np.float64)

        offset = 0
        for k in range(1, N + 1):
            if k == 1:
                offset += 1
                continue

            A_sub = np.zeros((k, k), dtype=np.float64)
            for i in range(k):
                gi = offset + i
                for j in range(k):
                    gj = offset + j
                    if i == j and self.self_loops:
                        A_sub[i, j] = (-1) ** gi
                    elif i != j:
                        A_sub[i, j] = (-1) ** (gi + gj)

            if k == 2:
                desired_r = self.spectral_radius
            else:
                desired_r = (N - k + 1) * self.spectral_radius / N

            current_r = spectral_radius_hybrid(A_sub)
            if current_r > 1e-14 and desired_r != 0:
                A_sub *= desired_r / current_r

            A[offset:offset + k, offset:offset + k] = A_sub
            offset += k

        A = np.triu(A) + np.triu(A, 1).T
        return A.astype(np.float64)

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "self_loops": self.self_loops,
            "spectral_radius": self.spectral_radius,
        }
        config.update(base_config)
        return config

