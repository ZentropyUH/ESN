from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import tensorflow as tf

from scipy.linalg import eigvals
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

from keras_reservoir_computing.utils.general import create_rng


def to_tensor(graph_func: Callable) -> Callable:
    """
    Decorator that converts a NetworkX graph returned by the `graph_func` into a
    2D TensorFlow tensor representing its adjacency matrix.

    Parameters
    ----------
    graph_func : callable
        A function that returns a NetworkX graph.

    Returns
    -------
    callable
        A wrapper function that, when called, returns a 2D TensorFlow tensor
        (float32 dtype) corresponding to the adjacency matrix of the graph.

    Notes
    -----
    - The original `graph_func` should return a NetworkX `Graph` or `DiGraph`.
    - The wrapper converts that graph into a NumPy array via `nx.to_numpy_array()`,
      then into a TensorFlow tensor using `tf.convert_to_tensor()`.
    """

    def wrapper(*args, **kwargs) -> tf.Tensor:
        G = graph_func(*args, **kwargs)  # Generate the graph
        adj_matrix = nx.to_numpy_array(G, nodelist=G.nodes, dtype=np.float32)
        tensor = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        return tensor

    return wrapper


def connected_graph(graph_func: Callable) -> Callable:
    """
    Decorator that ensures the generated graph adjacency matrix is connected
    (in the sense of strong connectivity for directed graphs).

    Parameters
    ----------
    graph_func : callable
        A function that generates a graph adjacency matrix (either as a tf.Tensor or np.ndarray).
        The function signature is expected to be:
            (n: int, *args, seed=None, tries=100, **kwargs)
        where:
            - n is the number of nodes,
            - seed is the random seed or RNG,
            - tries is the maximum number of attempts to generate a connected graph,
            - *args and **kwargs are other parameters.

    Returns
    -------
    callable
        A wrapper function that keeps generating a graph until it is connected or
        until the maximum number of tries is reached.

    Raises
    ------
    ValueError
        If a connected graph cannot be generated within the specified number of tries.

    Notes
    -----
    - Internally uses `scipy.sparse.csgraph.connected_components` to check connectivity.
    - For directed graphs, it uses `connection="strong"` to require strong connectivity.
    - The decorated function should return an NxN adjacency in either tf.Tensor or np.ndarray form.
    - `graph_func` may produce either directed or undirected adjacency matrices,
      controlled by its internal logic or kwargs (like `directed=True`).
    """

    def wrapper(
        n: int,
        *args,
        seed: Optional[Union[int, np.random.Generator]] = None,
        tries: int = 100,
        **kwargs,
    ) -> Union[np.ndarray, tf.Tensor]:
        rng = create_rng(seed)
        for _ in range(tries):
            # Generate adjacency matrix as tf.Tensor or np.ndarray
            G = graph_func(n, *args, seed=rng, **kwargs)

            # Convert to NumPy array if necessary for connected_components
            if isinstance(G, tf.Tensor):
                G = G.numpy()

            # Ensure G is 2D NxN
            if len(G.shape) != 2 or G.shape[0] != G.shape[1]:
                raise ValueError(
                    f"Adjacency matrix must be 2D NxN, got shape {G.shape}."
                )

            # Check connectivity
            n_components = connected_components(
                G,
                directed=kwargs.get("directed", True),
                connection="strong",
                return_labels=False,
            )

            if n_components == 1:
                return G

        raise ValueError(f"Could not generate a connected graph after {tries} tries.")

    return wrapper


def spectral_radius_hybrid(A: Union[tf.Tensor, np.ndarray]) -> float:
    """
    Computes the spectral radius (largest absolute eigenvalue) of matrix A,
    switching between sparse and dense methods depending on sparsity.

    Parameters
    ----------
    A : tf.Tensor or np.ndarray
        The input square matrix of shape (N, N). If A is a tf.Tensor, it is converted to
        a NumPy array internally.

    Returns
    -------
    float
        The absolute value of the largest eigenvalue of A.

    Notes
    -----
    - If the proportion of non-zero entries is below 50%, the matrix is treated
      as sparse and uses `scipy.sparse.linalg.eigs`.
    - Otherwise, `scipy.linalg.eigvals` is used.
    - The function returns a Python float (not a tf.Tensor).

    Raises
    ------
    ValueError
        If `A` is not 2D or not square.
    """
    # Convert A to tf.Tensor (float32) if it isn't already
    A = tf.convert_to_tensor(A, dtype=tf.float32)

    # Check dimensions
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be 2D and square, got shape {A.shape}.")

    # Compute sparsity
    non_zero_count = tf.math.count_nonzero(A, dtype=tf.int32)
    total_elements = tf.size(A)
    sparsity_ratio = tf.cast(non_zero_count, tf.float32) / tf.cast(
        total_elements, tf.float32
    )

    # Convert to NumPy for SciPy calls
    A_np = A.numpy()

    if sparsity_ratio < 0.5:
        # Use sparse eigenvalue calculation
        A_sparse = coo_matrix(A_np)
        # k=1 -> largest eigenvalue in magnitude
        # v0 -> initial vector for iteration
        val = eigs(
            A_sparse,
            k=1,
            which="LM",
            return_eigenvectors=False,
            v0=np.ones(A.shape[0], dtype=np.float32),
        )[0]
        s_radius = abs(val)
    else:
        # Use dense eigenvalue calculation
        s_radius = np.max(np.abs(eigvals(A_np)))

    return float(s_radius)
