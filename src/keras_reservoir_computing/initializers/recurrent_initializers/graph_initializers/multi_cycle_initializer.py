import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import multi_cycle


@tf.keras.utils.register_keras_serializable(package="krc", name="MultiCycleGraphInitializer")
class MultiCycleGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of multi-cycle graphs.

    Generates a multi-cycle graph.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    k : int
        Number of cycles in the graph.
    weight : float
        Weight of the edges in the graph.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.

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
        k: int = 3,
        weight: float = 1.0,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.k = k
        self.weight = weight
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
            A 2D adjacency matrix representing the generated multi-cycle graph.
        """
        adj = multi_cycle(
            n=n,
            k=self.k,
            weight=self.weight,
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
            "k": self.k,
            "weight": self.weight,
        }

        config.update(base_config)
        return config
