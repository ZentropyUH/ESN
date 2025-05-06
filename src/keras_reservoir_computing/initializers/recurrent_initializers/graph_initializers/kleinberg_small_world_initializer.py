import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import kleinberg_small_world


@tf.keras.utils.register_keras_serializable(
    package="krc", name="KleinbergSmallWorldGraphInitializer"
)
class KleinbergSmallWorldGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of Kleinberg's small-world graphs.

    Generates a graph where long-range connections follow a probability distribution
    based on distance.

    Parameters
    ----------
    q : float, optional
        Exponent controlling long-range connection probability decay.
    k : int, optional
        Number of long-range connections per node.
    directed : bool, optional
        If True, generates a directed graph.
    weighted : bool, optional
        If True, assigns weights to edges based on distance.
    beta : float, optional
        Exponent for weight calculation if `weighted=True`.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a Kleinberg small-world graph.

    Notes
    -----
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1 if `weighted=False`. Otherwise, the weights are generated proportional to the distance to the power of `beta`.
    """
    def __init__(
        self,
        q: float = 2,
        k: int = 1,
        directed: bool = True,
        weighted: bool = False,
        beta: float = 2,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.q = q
        self.k = k
        self.directed = directed
        self.weighted = weighted
        self.beta = beta
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:
        """
        Generate the adjacency matrix for a Kleinberg small-world graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated Kleinberg small-world graph.
        """
        adj = kleinberg_small_world(
            n=n,
            q=self.q,
            k=self.k,
            directed=self.directed,
            weighted=self.weighted,
            beta=self.beta,
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
            "q": self.q,
            "k": self.k,
            "directed": self.directed,
            "weighted": self.weighted,
            "beta": self.beta,
        }

        config.update(base_config)
        return config

