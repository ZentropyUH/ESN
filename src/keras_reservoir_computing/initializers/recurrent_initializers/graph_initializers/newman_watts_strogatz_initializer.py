import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import newman_watts_strogatz


@tf.keras.utils.register_keras_serializable(
    package="krc", name="NewmanWattsStrogatzGraphInitializer"
)
class NewmanWattsStrogatzGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of Newman-Watts-Strogatz small-world graphs.

    Generates a connected small-world graph with additional randomly rewired edges.

    Parameters
    ----------
    k : int
        Each node is initially connected to `k/2` predecessors and `k/2` successors.
    p : float
        Probability of adding new random edges.
    directed : bool, optional
        If True, generates a directed graph.
    self_loops : bool, optional
        If True, allows self-loops.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a Newman-Watts-Strogatz graph.

    Notes
    -----
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1.
    """

    def __init__(
        self,
        k: int = 4,
        p: float = 0.5,
        directed: bool = True,
        self_loops: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.k = k
        self.p = p
        self.directed = directed
        self.self_loops = self_loops
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
        """
        Generate the adjacency matrix for a Newman-Watts-Strogatz graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated Newman-Watts-Strogatz graph.
        """
        adj = newman_watts_strogatz(
            n=n,
            k=self.k,
            p=self.p,
            directed=self.directed,
            self_loops=self.self_loops,
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
            "k": self.k,
            "p": self.p,
            "directed": self.directed,
            "self_loops": self.self_loops,
        }

        config.update(base_config)
        return config
