from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import tensorflow as tf

from scipy.linalg import eigvals
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs, ArpackNoConvergence

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
        The input square matrix of shape (N, N).

    Returns
    -------
    float
        The absolute value of the largest eigenvalue of A.

    Raises
    ------
    TypeError
        If `A` is not a TF tensor or NumPy array.
    ValueError
        If `A` is not 2D or not square.
    """
    # get a NumPy view
    if isinstance(A, tf.Tensor):
        A_np = A.numpy()
    elif isinstance(A, np.ndarray):
        A_np = A
    else:
        raise TypeError(f"A must be tf.Tensor or np.ndarray, got {type(A)}")

    # shape checks
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        raise ValueError(f"A must be 2D and square, got shape {A_np.shape}")

    # sparsity
    nnz = np.count_nonzero(A_np)
    total = A_np.size
    if nnz / total < 0.5:
        # sparse path
        A_sp = coo_matrix(A_np)
        try:
            val = eigs(
                A_sp,
                k=1,
                which="LM",
                return_eigenvectors=False,
                v0=np.ones(A_np.shape[0], dtype=A_np.dtype),
            )[0]
            radius = abs(val)
        except ArpackNoConvergence as e:
            # use any converged values, else fallback to dense
            ev = getattr(e, "eigenvalues", None)
            if ev is not None and ev.size > 0:
                radius = np.max(np.abs(ev))
            else:
                radius = np.max(np.abs(eigvals(A_np)))
    else:
        # dense path
        radius = np.max(np.abs(eigvals(A_np)))

    return float(radius)
