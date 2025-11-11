from typing import Iterable, Optional, Union

import tensorflow as tf

from .base import GraphInitializerBase
from .graph_generators import dendrocycle_with_chords


@tf.keras.utils.register_keras_serializable(
    package="krc", name="ChordDendrocycleGraphInitializer"
)
class ChordDendrocycleGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of chord dendrocycles.

    Generates a chord dendrocycle graph with a core cycle and dendritic chains. Optionally, small-world chords are added to the core ring.

    Parameters
    ----------
    c : float
        Fraction for core cycle. (0 < c <= 1)
    d : float
        Fraction for dendrites. 0 <= d and c + d <= 1
    core_weight: float
        Single weight for all core edges.
    dendritic_weight: float
        Single weight for all dendritic edges.
    quiescent_weight: float
        Single weight for all quiescent edges.
    L : int or iterable of ints, optional
        Delays for backward chords. If None, no chords are added.
    w : float
        Base chord weight.
    alpha : float
        Geometric decay for multiple chord lengths.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int, optional
        RNG for DAG structure.

    Returns
    -------
    Tensor
        A 2D adjacency matrix of a chord dendrocycle graph.

    """
    def __init__(
        self,
        c: float=0.5,
        d: float=0.5,
        core_weight: float = 1.0,
        dendritic_weight: float = 1.0,
        quiescent_weight: float = 1.0,
        L: Optional[Union[int, Iterable[int]]] = None,
        w: float = 0.5,
        alpha: float = 1.0,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.c = c
        self.d = d
        self.core_weight = core_weight
        self.dendritic_weight = dendritic_weight
        self.quiescent_weight = quiescent_weight
        self.L = L
        self.w = w
        self.alpha = alpha
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(
        self,
        n: int
    ) -> tf.Tensor:
        """
        Generate the adjacency matrix for a chord dendrocycle graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated chord dendrocycle graph.
        """
        adj = dendrocycle_with_chords(
            n=n,
            c=self.c,
            d=self.d,
            core_weight=self.core_weight,
            dendritic_weight=self.dendritic_weight,
            quiescent_weight=self.quiescent_weight,
            L=self.L,
            w=self.w,
            alpha=self.alpha,
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
            "L": self.L,
            "w": self.w,
            "alpha": self.alpha,
        }

        config.update(base_config)
        return config

