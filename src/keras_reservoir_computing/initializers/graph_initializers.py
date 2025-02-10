from typing import Optional

import keras
import numpy as np
import tensorflow as tf
from keras.src.initializers import Initializer

from keras_reservoir_computing.utils.general_utils import create_rng
from keras_reservoir_computing.utils.graph_utils import (
    barabasi_albert,
    complete,
    connected_erdos_renyi,
    connected_watts_strogatz,
    kleinberg_small_world,
    newman_watts_strogatz,
    regular,
)
from keras_reservoir_computing.utils.graph_utils.helpers import spectral_radius_hybrid


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="GraphInitializerBase"
)
class GraphInitializerBase(Initializer):
    """Initializer that generates adjacency matrices according to a given function.

    Parameters
    ----------
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of a graph.

    Notes
    -----
    - This is a base class that should be inherited and the `_generate_adjacency_matrix` method should be implemented.
    - The _generate_adjacency_matrix method should return a 2D numpy array or a 2D tf.Tensor.
    """

    def __init__(
        self,
        spectral_radius: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if spectral_radius is not None and spectral_radius < 0:
            raise ValueError("The spectral radius should be non-negative.")

        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = create_rng(seed)
        super().__init__()

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        if isinstance(shape, int):
            shape = (shape, shape)
        elif (not isinstance(shape, (tuple, list)) or len(shape) != 2) or shape[
            0
        ] != shape[1]:
            raise ValueError("The shape of the adjacency matrix should be 2D.")

        n = shape[0]

        # Here is the BANANA
        adj = self._generate_adjacency_matrix(n)

        if self.spectral_radius is not None:
            sr = spectral_radius_hybrid(adj)
            adj = adj * self.spectral_radius / sr

        # Return the adjacency matrix as a 2D tensor
        return tf.convert_to_tensor(adj, dtype=dtype)

    def _generate_adjacency_matrix(self, n: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "The adjacency matrix generation function is not implemented."
        )

    def get_config(self):
        base_config = super().get_config()

        config = {
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }  # seed and spectral_radius are handled here
        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="WattsStrogatzGraphInitializer"
)
class WattsStrogatzGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of Watts-Strogatz graphs.

    Parameters
    ----------
    k : int
        The number of nearest neighbors.
    p : float
        The rewiring probability.
    directed : bool
        If True, the generated graph is directed.
    self_loops : bool
        If True, the generated graph has self-loops.
    tries : int
        The maximum number of tries to generate a connected graph.
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of a connected Watts-Strogatz graph.
    """

    def __init__(
        self,
        k: int=4,
        p: float=0.5,
        directed: bool = True,
        self_loops: bool = True,
        tries: int = 100,
        spectral_radius: float = None,
        seed: int = None,
    ):
        self.k = k
        self.p = p
        self.directed = directed
        self.self_loops = self_loops
        self.tries = tries
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
        adj = connected_watts_strogatz(
            n=n,
            k=self.k,
            p=self.p,
            directed=self.directed,
            self_loops=self.self_loops,
            tries=self.tries,
            seed=self.rng,
        )
        return adj

    def get_config(self):
        base_config = super().get_config()
        config = {
            "k": self.k,
            "p": self.p,
            "directed": self.directed,
            "self_loops": self.self_loops,
            "tries": self.tries,
        }  # seed and spectral_radius are handled by the base class

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="ErdosRenyiGraphInitializer"
)
class ErdosRenyiGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of Erdos-Renyi graphs.

    Parameters
    ----------
    p : float
        The edge probability.
    directed : bool
        If True, the generated graph is directed.
    self_loops : bool
        If True, the generated graph has self-loops.
    tries : int
        The maximum number of tries to generate a connected graph.
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of an Erdos-Renyi graph.
    """

    def __init__(
        self,
        p: float=0.5,
        directed: bool = True,
        self_loops: bool = True,
        tries: int = 100,
        spectral_radius: float = None,
        seed: int = None,
    ):
        self.p = p
        self.directed = directed
        self.self_loops = self_loops
        self.tries = tries
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
        adj = connected_erdos_renyi(
            n=n,
            p=self.p,
            directed=self.directed,
            self_loops=self.self_loops,
            tries=self.tries,
            seed=self.rng,
        )
        return adj

    def get_config(self):
        base_config = super().get_config()
        config = {
            "p": self.p,
            "directed": self.directed,
            "self_loops": self.self_loops,
            "tries": self.tries,
        }  # seed and spectral_radius are handled by the base class

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="BarabasiAlbertGraphInitializer"
)
class BarabasiAlbertGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of Barabasi-Albert graphs.

    Parameters
    ----------
    m : int
        The number of edges to attach from a new node to existing nodes.
    directed : bool
        If True, the generated graph is directed.
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of a Barabasi-Albert graph.
    """

    def __init__(
        self,
        m: int=3,
        directed: bool = True,

        spectral_radius: float = None,
        seed: int = None,
    ):
        self.m = m
        self.directed = directed
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
        adj = barabasi_albert(
            n=n,
            m=self.m,
            directed=self.directed,
            seed=self.rng,
        )
        return adj

    def get_config(self):
        base_config = super().get_config()
        config = {
            "m": self.m,
            "directed": self.directed,
        }

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="NewmanWattsStrogatzGraphInitializer"
)
class NewmanWattsStrogatzGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of Newman-Watts-Strogatz graphs.

    Parameters
    ----------
    k : int
        Each node initially connects to k/2 predecessors and k/2 successors.
    p : float
        Rewiring probability.
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected one.
    self_loops : bool, optional
        If True, allows self-loops during rewiring.
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of a connected Newman-Watts-Strogatz graph.
    """

    def __init__(
        self,
        k: int=4,
        p: float=0.5,
        directed: bool = True,
        self_loops: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ):
        self.k = k
        self.p = p
        self.directed = directed
        self.self_loops = self_loops
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
        adj = newman_watts_strogatz(
            n=n,
            k=self.k,
            p=self.p,
            directed=self.directed,
            self_loops=self.self_loops,
            seed=self.rng,
        )
        return adj

    def get_config(self):
        base_config = super().get_config()
        config = {
            "k": self.k,
            "p": self.p,
            "directed": self.directed,
            "self_loops": self.self_loops,
        }
        
        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="KleinbergSmallWorldGraphInitializer"
)
class KleinbergSmallWorldGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of Kleinberg's small-world graphs.

        Parameters
        ----------
        q : float, optional
            Exponent controlling long-range connection probability decay (default: 2).
        k : int, optional
            Number of long-range connections per node (default: 1).
        directed : bool, optional
            If True, generates a directed graph; otherwise, undirected (default: False).
        weighted : bool, optional
            If True, assigns weights proportional to distance^beta (default: False).
        beta : float, optional
            Exponent for weight calculation when weighted=True (default: 2).
        spectral_radius : float or None
            The spectral radius of the generated graph. If None, the spectral radius is not controlled.
        seed : int
            The random seed for reproducibility. If None, the random seed is not set.
        Returns
        -------
        tf.Tensor
            An initialized 2D tensor as an adjacency matrix of a Kleinberg's small-world graph.
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
    ):
        self.q = q
        self.k = k
        self.directed = directed
        self.weighted = weighted
        self.beta = beta
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
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

    def get_config(self):
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


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="RegularGraphInitializer"
)
class RegularGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of regular graphs.

    Parameters
    ----------
    k : int
        The degree of the graph.
    directed : bool
        If True, the generated graph is directed.
    self_loops : bool
        If True, the generated graph has self-loops.
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of a regular graph.
    """

    def __init__(
        self,
        k: int=2,
        directed: bool = True,
        self_loops: bool = True,
        random_weights: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ):
        self.k = k
        self.directed = directed
        self.self_loops = self_loops
        self.random_weights = random_weights
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
        adj = regular(
            n=n,
            k=self.k,
            directed=self.directed,
            self_loops=self.self_loops,
            random_weights=self.random_weights,
            seed=self.rng,
        )
        return adj

    def get_config(self):
        base_config = super().get_config()
        config = {
            "k": self.k,
            "directed": self.directed,
            "self_loops": self.self_loops,
        }

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(
    package="MyInitializers", name="CompleteGraphInitializer"
)
class CompleteGraphInitializer(GraphInitializerBase):
    """Initializer that generates adjacency matrices of complete graphs.

    Parameters
    ----------
    self_loops : bool
        If True, the generated graph has self-loops.
    random_weights : bool
        If True, the generated graph has random weights. If False, the weights will be alternating between 1 and -1.
    spectral_radius : float or None
        The spectral radius of the generated graph. If None, the spectral radius is not controlled.
    seed : int
        The random seed for reproducibility. If None, the random seed is not set.
    Returns
    -------
    tf.Tensor
        An initialized 2D tensor as an adjacency matrix of a complete graph.
    """

    def __init__(
        self,
        self_loops: bool = True,
        random_weights: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ):
        self.self_loops = self_loops
        self.random_weights = random_weights
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> np.ndarray:
        adj = complete(
            n=n,
            self_loops=self.self_loops,
        )
        return adj

    def get_config(self):
        base_config = super().get_config()
        config = {
            "self_loops": self.self_loops,
            "random_weights": self.random_weights,
        }

        config.update(base_config)
        return config


__all__ = [
    "WattsStrogatzGraphInitializer",
    "ErdosRenyiGraphInitializer",
    "BarabasiAlbertGraphInitializer",
    "NewmanWattsStrogatzGraphInitializer",
    "KleinbergSmallWorldGraphInitializer",
    "RegularGraphInitializer",
    "CompleteGraphInitializer",
]

def __dir__():
    return __all__
