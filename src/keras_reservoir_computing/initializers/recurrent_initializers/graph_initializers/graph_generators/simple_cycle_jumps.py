from networkx import DiGraph

from keras_reservoir_computing.initializers.helpers import (
    to_tensor,
)


@to_tensor
def simple_cycle_jumps(
    n: int, 
    l: int, 
    r_c: float = 1.0, 
    r_l: float = 0.5
    ) -> DiGraph:
    """
    Generate a directed cycle with bidirectional jumps.

    Parameters
    ----------
    n : int
        Total number of nodes.
    l : int
        Jump step size.
    r_c : float
        Weight for cycle edges.
    r_l : float
        Weight for jump edges.

    Returns
    -------
    G : nx.DiGraph
        Directed graph with:
        - A directed cycle of n nodes (weight = r_c)
        - Bidirectional jump edges every l nodes (weight = r_l)
          until n - (n % l).
    """
    G = DiGraph()
    G.add_nodes_from(range(n))

    # Directed cycle
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=r_c)

    # Bidirectional jumps
    limit = n - (n % l)
    for i in range(0, limit, l):
        j = (i + l) % n
        G.add_edge(i, j, weight=r_l)
        G.add_edge(j, i, weight=r_l)

    return G
