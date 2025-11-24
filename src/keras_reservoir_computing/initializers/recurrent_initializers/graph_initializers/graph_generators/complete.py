from typing import Optional, Union

import numpy as np
from networkx import DiGraph, Graph

from keras_reservoir_computing.initializers.helpers import (
    create_rng,
    to_tensor,
)


@to_tensor
def complete(
    n: int,
    self_loops: bool = False,
    random_weights: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a complete (undirected) graph of n nodes and returns its adjacency matrix as a Tensor.

    Each pair of distinct nodes is connected by an edge. Optionally, self-loops can be included.
    Weights on edges can be random in {-1, 1} or follow a deterministic alternating pattern.

    Parameters
    ----------
    n : int
        Number of nodes.
    self_loops : bool, optional
        If True, adds a self-loop to each node. Default: False.
    random_weights : bool, optional
        If True, weights are chosen randomly from {-1, 1}; otherwise, they alternate
        according to (-1)^(i + j). Default: True.
    seed : int or np.random.Generator or None, optional
        Seed for the RNG.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the complete graph.
    """
    rng = create_rng(seed)
    G = Graph()  # Always undirected in this function

    for i in range(n):
        for j in range(i + 1, n):
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** (i + j)
            G.add_edge(i, j, weight=weight)

    if self_loops:
        for i in range(n):
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** i
            G.add_edge(i, i, weight=weight)

    return G
