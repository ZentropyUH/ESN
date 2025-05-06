import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import complete


@tf.keras.utils.register_keras_serializable(
    package="krc", name="CompleteGraphInitializer"
)
class CompleteGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of complete graphs.

    Generates a fully connected graph where every node is connected to every other node.

    Parameters
    ----------
    self_loops : bool, optional
        If True, allows self-loops.
    random_weights : bool, optional
        If True, assigns random weights to edges. If False, weights alternate between 1 and -1.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a complete graph.

    Notes
    -----
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1 if `random_weights=False`. Otherwise, the weights are alternated between -1 and 1.
    - This is equivalent to a dense matrix with alternating -1 and 1 values.
    """
    def __init__(
        self,
        self_loops: bool = True,
        random_weights: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.self_loops = self_loops
        self.random_weights = random_weights
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:
        """
        Generate the adjacency matrix for a complete graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated complete graph.
        """
        adj = complete(
            n=n,
            self_loops=self.self_loops,
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
            "self_loops": self.self_loops,
            "random_weights": self.random_weights,
        }

        config.update(base_config)
        return config

