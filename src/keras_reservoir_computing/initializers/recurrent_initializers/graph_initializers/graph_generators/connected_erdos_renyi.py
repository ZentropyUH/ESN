from typing import Optional, Union

import numpy as np
from networkx import DiGraph, Graph

from keras_reservoir_computing.initializers.helpers import (
    connected_graph,
)

from .erdos_renyi import erdos_renyi


@connected_graph
def connected_erdos_renyi(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a **connected** Erdos-Renyi graph and returns its adjacency matrix as a Tensor.

    This function wraps :func:`erdos_renyi` with a decorator that attempts multiple
    generations until a connected graph is obtained (up to a certain number of tries).
    Then, the resulting graph is converted to a Tensor adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of including each edge (in [0, 1]).
    directed : bool
        If True, generates a directed graph; otherwise, an undirected graph.
    self_loops : bool
        If True, allows self-loops in the graph.
    seed : int or np.random.Generator or None
        Seed for the random number generator.
    tries : int, optional
        Number of attempts to generate a connected graph. This parameter is handled
        by the ``@connected_graph`` decorator.

    Returns
    -------
    tf.Tensor
        Adjacency matrix (as a Tensor) of the connected Erdos-Renyi graph.

    Raises
    ------
    ValueError
        If the probability `p` is too small to expect a connected graph. As a rough guideline,
        `p` should be greater than `ln(n)/n` for a good chance of connectivity.
    """
    if p < np.log(n) / n:
        raise ValueError(
            f"Edge probability p must be > ln(n) / n to have a good chance of connectivity. "
            f"(Got p={p}, n={n})."
        )
    return erdos_renyi(n, p, directed, self_loops, seed)
