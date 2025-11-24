from typing import Optional, Union

import numpy as np
from networkx import DiGraph, Graph

from keras_reservoir_computing.initializers.helpers import (
    create_rng,
    to_tensor,
)


@to_tensor
def barabasi_albert(
    n: int,
    m: int,
    directed: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a Barabási-Albert scale-free network and returns its adjacency matrix as a Tensor.

    The Barabási-Albert model grows a graph one node at a time, linking the new node to
    ``m`` existing nodes with probability proportional to their degrees.

    .. note::
        This function is decorated with ``@to_tensor``, so it returns the adjacency matrix
        as a Tensor.

    Parameters
    ----------
    n : int
        Total number of nodes in the final graph.
    m : int
        Number of edges each new node creates with already existing nodes.
        Must be >= 1 and < n.
    directed : bool, optional
        If True, creates a directed scale-free network; otherwise, undirected.
    seed : int or np.random.Generator or None, optional
        Seed for the RNG.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the Barabási-Albert graph.

    Raises
    ------
    ValueError
        If ``m < 1 or m >= n``.
    """
    if m < 1 or m >= n:
        raise ValueError(f"m must be >= 1 and < n, got m={m}, n={n}.")

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initialize a complete graph with m nodes
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(j, i, weight=rng.choice([-1, 1]))

    # Keep track of node 'targets' with frequency proportional to node degree
    targets = list(G.nodes) * m

    # Add remaining nodes
    for i in range(m, n):
        G.add_node(i)
        new_edges = rng.choice(targets, size=m, replace=False)
        for t in new_edges:
            G.add_edge(i, t, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(t, i, weight=rng.choice([-1, 1]))

        # Update 'targets' to reflect new degrees
        targets.extend([i] * m)
        targets.extend(new_edges)

    return G
