import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import dendrocycle


@tf.keras.utils.register_keras_serializable(
    package="krc", name="DendrocycleGraphInitializer"
)
class DendrocycleGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of dendro-cycles.

    Generates a dendro-cycle graph.

    Parameters
    ----------
    c : float
        Fraction of nodes in the core cycle (0 < c < 1).
    d : float
        Fraction of nodes in dendrites (0 <= d < 1, c + d <= 1).
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a dendro-cycle graph.

    Notes
    -----
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1.
    """
    def __init__(
        self,
        c: float=0.5,
        d: float=0.5,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.c = c
        self.d = d
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:
        """
        Generate the adjacency matrix for a dendro-cycle graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated dendro-cycle graph.
        """
        adj = dendrocycle(
            n=n,
            c=self.c,
            d=self.d,
            seed=self.rng,
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
            "c": self.c,
            "d": self.d,
            "directed": self.directed,
        }

        config.update(base_config)
        return config

