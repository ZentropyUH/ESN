import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import connected_erdos_renyi


@tf.keras.utils.register_keras_serializable(package="krc", name="ErdosRenyiGraphInitializer")
class ErdosRenyiGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of Erdos-Renyi random graphs.

    Generates a connected Erdos-Renyi graph where edges are added with a given probability.

    Parameters
    ----------
    p : float
        Probability of an edge existing between any two nodes.
    directed : bool, optional
        If True, generates a directed graph.
    self_loops : bool, optional
        If True, allows self-loops.
    tries : int, optional
        Maximum attempts to ensure graph connectivity.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of an Erdos-Renyi graph.

    Notes
    -----
    - The generated graph is guaranteed to be connected.
    - The number of tries to ensure connectivity can be adjusted with the `tries` parameter.
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1.
    """

    def __init__(
        self,
        p: float = 0.5,
        directed: bool = True,
        self_loops: bool = True,
        tries: int = 100,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.p = p
        self.directed = directed
        self.self_loops = self_loops
        self.tries = tries
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
        """
        Generate the adjacency matrix for a connected Erdos-Renyi graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated Erdos-Renyi graph.
        """
        adj = connected_erdos_renyi(
            n=n,
            p=self.p,
            directed=self.directed,
            self_loops=self.self_loops,
            tries=self.tries,
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
            "p": self.p,
            "directed": self.directed,
            "self_loops": self.self_loops,
            "tries": self.tries,
        }  # seed and spectral_radius are handled by the base class

        config.update(base_config)
        return config
