import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import barabasi_albert


@tf.keras.utils.register_keras_serializable(package="krc", name="BarabasiAlbertGraphInitializer")
class BarabasiAlbertGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of Barabasi-Albert scale-free graphs.

    Generates a Barabasi-Albert graph using preferential attachment.

    Parameters
    ----------
    m : int
        Number of edges each new node forms with existing nodes.
    directed : bool, optional
        If True, generates a directed graph.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a Barabasi-Albert graph.

    Notes
    -----
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1.
    """

    def __init__(
        self,
        m: int = 3,
        directed: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.m = m
        self.directed = directed
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
        """
        Generate the adjacency matrix for a Barabasi-Albert graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated Barabasi-Albert graph.
        """
        adj = barabasi_albert(
            n=n,
            m=self.m,
            directed=self.directed,
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
            "m": self.m,
            "directed": self.directed,
        }

        config.update(base_config)
        return config
