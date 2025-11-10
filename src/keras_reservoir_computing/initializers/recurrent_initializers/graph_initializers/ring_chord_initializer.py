import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import ring_chord


@tf.keras.utils.register_keras_serializable(
    package="krc", name="RingChordGraphInitializer"
)
class RingChordGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of ring-chord graphs.

    Generates a graph with a ring undirected topology and long jumps.

    Parameters
    ----------
    L: int
        Long jump (1 ≤ L ≤ N//2). If L=1, weights accumulate with local edges.
    w: float
        Weight of long edges (>= 0).
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a ring-chord graph.
    """
    def __init__(
        self,
        L: int,
        w: float,
        alpha: float = 1.0,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.L = L
        self.w = w
        self.alpha = alpha
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:
        """
        Generate the adjacency matrix for a ring-chord graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated ring-chord graph.
        """
        adj = ring_chord(
            n=n,
            L=self.L,
            w=self.w,
            alpha=self.alpha,
        )
        return adj

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        base_config = super().get_config()
        config = {
            "L": self.L,
            "w": self.w,
            "alpha": self.alpha,
        }

        config.update(base_config)
        return config

