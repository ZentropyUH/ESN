from typing import Optional, Union

import numpy as np
from networkx import DiGraph, Graph

from keras_reservoir_computing.initializers.helpers import (
    create_rng,
    to_tensor,
)


@to_tensor
def newman_watts_strogatz(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a Newman-Watts-Strogatz small-world graph and returns its adjacency matrix as a Tensor.

    Similar to Watts-Strogatz, except existing edges are **not removed** during rewiring.
    Instead, new edges are added with probability ``p``.

    .. note::
        Since the function is decorated with ``@to_tensor``, the returned value is a
        Tensor adjacency matrix, not a NetworkX graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to ``k/2`` predecessors and ``k/2`` successors.
        If ``k`` is odd, it will be incremented by 1 internally.
        Must be smaller than ``n``.
    p : float
        Probability of adding a long-range (random) edge.
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected graph.
    self_loops : bool, optional
        If True, allows self-loops during the additional edge creation.
    seed : int or np.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the Newman-Watts-Strogatz graph.

    Raises
    ------
    ValueError
        If ``k >= n``.
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n}).")

    if k % 2 != 0:
        k += 1

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initial ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(i, (i - j) % n, weight=rng.choice([-1, 1]))

    # Add edges with probability p (no edge removal)
    edges = list(G.edges())
    for u, v in edges:
        if rng.random() < p:
            candidates = rng.permutation(n)
            for new_v in candidates:
                if (new_v != u or self_loops) and not G.has_edge(u, new_v):
                    G.add_edge(u, new_v, weight=rng.choice([-1, 1]))
                    break

    return G
