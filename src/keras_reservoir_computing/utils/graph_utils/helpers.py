import networkx as nx
import numpy as np
import tensorflow as tf
from scipy.linalg import eigvals
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

from keras_reservoir_computing.utils.general_utils import create_rng


def to_tensor(graph_func: callable) -> callable:
    """
    Decorator that converts a NetworkX graph returned by the function into a 2D TensorFlow tensor.

    Parameters
    ----------
    func : callable
        Function that returns a NetworkX graph.

    Returns
    -------
    callable
        Wrapper function that converts the graph into a TensorFlow tensor.

    Notes
    -----
    - The function should return a NetworkX graph.
    - The wrapper function converts the graph into a NumPy array and then into a TensorFlow tensor.
    """

    def wrapper(*args, **kwargs):
        G = graph_func(*args, **kwargs)  # Generate the graph
        adj_matrix = nx.to_numpy_array(G, nodelist=G.nodes, dtype=np.float32)  # Convert to NumPy array
        tensor = tf.convert_to_tensor(adj_matrix)  # Convert to TensorFlow tensor
        return tensor

    return wrapper


def connected_graph(grap_func: callable) -> callable:
    """
    Decorator that ensures the generated graph is connected.

    Parameters
    ----------
    func : callable
        Function that generates a graph. This should return a 2D NxN tf.Tensor or a 2D NxN NumPy array.

    Returns
    -------
    callable
        Wrapper function that generates a connected graph.

    Notes
    -----
    - The function should return a 2D NxN tf.Tensor.
    - The graph_func should have a signature of (n, *args, seed=None, **kwargs).
        n (int): Number of nodes.
        *args: Additional arguments.
        seed (int, np.random.Generator, or None): Seed for the random number generator. Default is None.
        **kwargs: Additional keyword arguments.
    - The wrapper function generates a graph and checks if it is connected.
    - If the graph is not connected, it generates a new graph until a connected one is found or the maximum number of tries is reached.
    """

    def wrapper(n, *args, seed=None, tries=100, **kwargs):

        rng = create_rng(seed)

        for _ in range(tries):
            G = grap_func(n, *args, seed=rng, **kwargs)
            if (
                connected_components(
                    G,
                    directed=kwargs.get("directed", True),
                    connection="strong",
                    return_labels=False,
                )
                == 1
            ):
                return G

        raise ValueError(f"Could not generate a connected graph after {tries} tries.")

    return wrapper


def spectral_radius_hybrid(A: tf.Tensor) -> float:
    """
    Computes the spectral radius (largest eigenvalue) of a matrix, choosing between sparse and dense methods.

    Parameters
    ----------
    A : tf.Tensor or np.ndarray
        The input square matrix.

    Returns
    -------
    float
        The absolute value of the largest eigenvalue.

    Notes
    -----
    - Uses SciPy's `eigs` for sparse matrices (<50% non-zero elements).
    - Uses SciPy's `eigvals` for dense matrices.
    """
    A = tf.convert_to_tensor(A, dtype=tf.float32)

    # Compute sparsity using TensorFlow
    non_zero_count = tf.math.count_nonzero(A, dtype=tf.int32)
    total_elements = tf.size(A)
    sparsity_ratio = tf.cast(non_zero_count, tf.float32) / tf.cast(
        total_elements, tf.float32
    )

    if sparsity_ratio < 0.5:  # Treat as sparse
        A_sparse = coo_matrix(A.numpy())  # Convert to SciPy sparse format
        s_radius = abs(
            eigs(
                A_sparse,
                k=1,
                which="LM",
                return_eigenvectors=False,
                v0=np.ones(A.shape[0], dtype=np.float32),
            )[0]
        )
    else:  # Treat as dense
        s_radius = np.max(np.abs(eigvals(A)))  # General case

    return s_radius
