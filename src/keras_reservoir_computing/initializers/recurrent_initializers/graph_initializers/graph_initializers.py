from typing import List, Optional, Tuple, Union

import tensorflow as tf

from .generators import (
    barabasi_albert,
    complete,
    connected_erdos_renyi,
    connected_watts_strogatz,
    kleinberg_small_world,
    newman_watts_strogatz,
    regular,
)
from keras_reservoir_computing.initializers.helpers import (
    spectral_radius_hybrid,
)
from keras_reservoir_computing.utils.general import create_rng


@tf.keras.utils.register_keras_serializable(
    package="krc", name="GraphInitializerBase"
)
class GraphInitializerBase(tf.keras.Initializer):
    """
    Base class for initializers generating adjacency matrices for graph-based models.

    This initializer constructs adjacency matrices based on a specified graph generation
    function. It allows for optional spectral radius control to adjust the eigenvalues
    of the generated matrix.

    Parameters
    ----------
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Methods
    -------
    __call__(shape, dtype=None)
        Generates an adjacency matrix with the specified shape.
    _generate_adjacency_matrix(n, *args, **kwargs)
        Abstract method for generating a graph adjacency matrix.
    get_config()
        Returns a dictionary of the initializer's configuration.

    Returns
    -------
    Tensor
        A 2D adjacency matrix representing the generated graph.

    Notes
    -----
    This is an abstract base class and must be subclassed with a specific graph
    generation function implemented in `_generate_adjacency_matrix`.
    """
    def __init__(
        self,
        spectral_radius: Optional[float] = None,
        seed: Union[int, tf.random.Generator, None] = None,
    ) -> None:
        if spectral_radius is not None and spectral_radius < 0:
            raise ValueError("The spectral radius should be non-negative.")

        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = create_rng(seed)
        super().__init__()

    def __call__(
        self,
        shape: Union[int, Tuple[int, int], List[int]],
        dtype: Optional[tf.dtypes.DType] = None,
    ) -> tf.Tensor:
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

    def _generate_adjacency_matrix(self, n: int, *args, **kwargs) -> tf.Tensor:
        """
        Abstract method for generating a graph adjacency matrix.

        This method should be implemented by subclasses to generate the adjacency matrix
        for a specific type of graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.
        *args : tuple
            Additional positional arguments for the graph generation function.
        **kwargs : dict
            Additional keyword arguments for the graph generation function.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated graph.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "The adjacency matrix generation function is not implemented."
        )

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
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }  # seed and spectral_radius are handled here
        config.update(base_config)
        return config


@tf.keras.utils.register_keras_serializable(
    package="krc", name="WattsStrogatzGraphInitializer"
)
class WattsStrogatzGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of Watts-Strogatz small-world graphs.

    Generates a connected Watts-Strogatz graph with adjustable rewiring probability.

    Parameters
    ----------
    k : int
        Number of nearest neighbors each node is initially connected to.
    p : float
        Probability of rewiring each edge.
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
        A 2D adjacency matrix of a Watts-Strogatz graph.

    Notes
    -----
    - The generated graph is guaranteed to be connected.
    - The number of tries to ensure connectivity can be adjusted with the `tries` parameter.
    - The non-zero elements of the adjacency matrix are sampled as -1 or 1.
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
    ) -> None:
        self.k = k
        self.p = p
        self.directed = directed
        self.self_loops = self_loops
        self.tries = tries
        super().__init__(spectral_radius=spectral_radius, seed=seed)

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
        """
        Generate the adjacency matrix for a connected Watts-Strogatz graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated Watts-Strogatz graph.
        """
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
            "tries": self.tries,
        }  # seed and spectral_radius are handled by the base class

        config.update(base_config)
        return config


@tf.keras.utils.register_keras_serializable(
    package="krc", name="ErdosRenyiGraphInitializer"
)
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
        p: float=0.5,
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


@tf.keras.utils.register_keras_serializable(
    package="krc", name="BarabasiAlbertGraphInitializer"
)
class BarabasiAlbertGraphInitializer(GraphInitializerBase):
    """
    Initializer for adjacency matrices of Barabasi-Albert scale-free graphs.

    Generates a Barabasi-Albert graph using preferential attachment.

    Parameters
    ----------
    m : int
        Number of edges each new node forms with existing nodes.
    directed : bool, optional
        If True, generates a directed graph.
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

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
        m: int=3,
        directed: bool = True,
        spectral_radius: float = None,
        seed: int = None,
    ) -> None:
        self.m = m
        self.directed = directed
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
            A 2D adjacency matrix representing the generated Barabasi-Albert graph.
        """
        adj = barabasi_albert(
            n=n,
            m=self.m,
            directed=self.directed,
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
            "m": self.m,
            "directed": self.directed,
        }

        config.update(base_config)
        return config


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
        k: int=4,
        p: float=0.5,
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

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
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

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
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

    def _generate_adjacency_matrix(self, n: int) -> tf.Tensor:
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
