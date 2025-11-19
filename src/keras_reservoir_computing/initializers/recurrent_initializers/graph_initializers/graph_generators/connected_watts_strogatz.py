from typing import Optional, Union

import numpy as np
from networkx import DiGraph, Graph

from keras_reservoir_computing.initializers.helpers import (
    connected_graph,
)

from .watts_strogatz import watts_strogatz


@connected_graph
def connected_watts_strogatz(
    n: int,
    k: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a **connected** Watts-Strogatz graph and returns its adjacency matrix as a Tensor.

    This function wraps :func:`watts_strogatz` with a decorator that attempts multiple
    generations until a connected graph is obtained (up to a certain number of tries).
    Then, the resulting graph is converted to a Tensor adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to ``k/2`` predecessors and ``k/2`` successors.
    p : float
        Rewiring probability in the interval [0, 1].
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected graph.
        Default is True.
    self_loops : bool, optional
        If True, allows self-loops during the rewiring step. Default is True.
    seed : int or np.random.Generator or None, optional
        Seed for random number generator (RNG). If None, a random seed is used.
    tries : int, optional
        Number of attempts to generate a connected graph. This parameter is handled
        by the ``@connected_graph`` decorator, not passed directly here.

    Returns
    -------
    tf.Tensor
        Adjacency matrix (as a Tensor) of the connected Watts-Strogatz graph.
    """
    return watts_strogatz(n, k, p, directed, self_loops, seed)
