import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import regular


@tf.keras.utils.register_keras_serializable(
    package="krc", name="RegularGraphInitializer"
)
class RegularGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of k-regular graphs.

    Generates a connected k-regular graph where each node has exactly `k` neighbors in a ring topology.

    Parameters
    ----------
    k : int
        Degree of each node.
    directed : bool, optional
        If True, generates a directed graph.
    self_loops : bool, optional
        If True, allows self-loops.
    random_weights : bool, optional
        If True, assigns random weights to edges.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a k-regular graph.

    Notes
    -----
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1 if `random_weights=False`. Otherwise, the weights are alternated between -1 and 1.
    """
    def __init__(
        self,
        k: int=2,
        directed: bool = True,
        self_loops: bool = True,
        random_weights: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.k = k
        self.directed = directed
        self.self_loops = self_loops
        self.random_weights = random_weights
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:
        """
        Generate the adjacency matrix for a k-regular graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated k-regular graph.
        """
        adj = regular(
            n=n,
            k=self.k,
            directed=self.directed,
            self_loops=self.self_loops,
            random_weights=self.random_weights,
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
            "directed": self.directed,
            "self_loops": self.self_loops,
        }

        config.update(base_config)
        return config

