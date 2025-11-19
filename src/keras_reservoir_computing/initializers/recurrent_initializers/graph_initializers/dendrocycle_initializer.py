import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import dendrocycle


@tf.keras.utils.register_keras_serializable(package="krc", name="DendrocycleGraphInitializer")
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
    core_weight: float
        Single weight for all core edges.
    dendritic_weight: float
        Single weight for all dendritic edges.
    quiescent_weight: float
        Single weight for all quiescent edges.
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
        c: float = 0.5,
        d: float = 0.5,
        core_weight: float = 1.0,
        dendritic_weight: float = 1.0,
        quiescent_weight: float = 1.0,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.c = c
        self.d = d
        self.core_weight = core_weight
        self.dendritic_weight = dendritic_weight
        self.quiescent_weight = quiescent_weight
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
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
            core_weight=self.core_weight,
            dendritic_weight=self.dendritic_weight,
            quiescent_weight=self.quiescent_weight,
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
            "core_weight": self.core_weight,
            "dendritic_weight": self.dendritic_weight,
            "quiescent_weight": self.quiescent_weight,
        }

        config.update(base_config)
        return config
