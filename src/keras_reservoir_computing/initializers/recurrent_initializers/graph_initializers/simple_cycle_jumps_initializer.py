import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import simple_cycle_jumps


@tf.keras.utils.register_keras_serializable(
    package="krc", name="SimpleCycleJumpsGraphInitializer"
)
class SimpleCycleJumpsGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of k-regular graphs.

    Generates a connected k-regular graph where each node has exactly `k` neighbors in a ring topology.

    Parameters
    ----------
    l : int
        Jump size. The number of nodes to skip for the jumps on the cycle.
    r_c : float
        The weight of the cycle edges.
    r_l :
        The weight of the jump edges.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
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
        l: int,
        r_c: float,
        r_l: float,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.l = l
        self.r_c = r_c
        self.r_l = r_l
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
        adj = simple_cycle_jumps(
            n=n,
            l=self.l,
            r_c=self.r_c,
            r_l=self.r_l
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
            "l": self.l,
            "r_c": self.r_c,
            "r_l": self.r_l,
        }

        config.update(base_config)
        return config

