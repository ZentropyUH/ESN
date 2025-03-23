from typing import Union, Optional
import numpy as np
from networkx import DiGraph, Graph

from keras_reservoir_computing.initializers.helpers import connected_graph, create_rng, to_tensor


@to_tensor
def watts_strogatz(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a Watts-Strogatz small-world graph and returns its adjacency matrix as a Tensor.

    The function starts by creating a ring lattice where each node is connected to ``k/2`` neighbors
    on each side. Then, with probability ``p``, each edge is rewired to a new node (allowing
    for possible self-loops if specified). Weights on edges are chosen randomly from the set ``{-1, 1}``.

    .. note::
        Since this function is decorated with ``@to_tensor``, the returned value is a Tensor
        of the adjacency matrix, *not* a NetworkX graph object.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is initially connected to ``k/2`` predecessors and ``k/2`` successors.
        If ``k`` is odd, it will be incremented by 1 internally.
        Must be smaller than ``n``.
    p : float
        Rewiring probability in the interval [0, 1].
    directed : bool, optional
        If True, generates a directed graph; otherwise, generates an undirected graph.
    self_loops : bool, optional
        If True, allows self-loops during the rewiring step.
    seed : int or np.random.Generator or None, optional
        Seed for random number generator (RNG). If None, a random seed is used.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the generated Watts-Strogatz graph, as a Tensor.

    Raises
    ------
    ValueError
        If ``k >= n`` (not a valid ring lattice).
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n}).")

    # Ensure k is even
    if k % 2 != 0:
        k += 1

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initial ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n, weight=rng.choice([-1, 1]))  # forward edge
            if directed:
                G.add_edge(i, (i - j) % n, weight=rng.choice([-1, 1]))  # backward edge

    # Rewire edges with probability p
    edges = list(G.edges())
    for u, v in edges:
        if rng.random() < p:
            # Remove the original edge
            G.remove_edge(u, v)

            # Find a new candidate node for rewiring
            candidates = rng.permutation(n)
            for new_v in candidates:
                if (new_v != u or self_loops) and not G.has_edge(u, new_v):
                    G.add_edge(u, new_v, weight=rng.choice([-1, 1]))
                    break

    return G


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


@to_tensor
def erdos_renyi(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates an Erdos-Renyi (G(n, p)) graph and returns its adjacency matrix as a Tensor.

    Every possible edge is included with probability ``p``, independently of every other edge.
    Weights on edges are chosen randomly from the set ``{-1, 1}``.

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

    Returns
    -------
    tf.Tensor
        Adjacency matrix (as a Tensor) of the generated Erdos-Renyi graph.
    """
    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    nodes = range(n)
    G.add_nodes_from(nodes)

    if directed:
        edges = [(u, v) for u in nodes for v in nodes if self_loops or u != v]
    else:
        # For undirected, only consider edges (u, v) with u <= v to avoid duplicates
        edges = [(u, v) for u in nodes for v in range(u, n) if self_loops or u != v]

    selected_edges = [edge for edge in edges if rng.random() < p]

    G.add_edges_from((u, v, {"weight": rng.choice([-1, 1])}) for u, v in selected_edges)

    return G


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

    rng = np.random.default_rng(seed)
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


@to_tensor
def kleinberg_small_world(
    n: int,
    q: float = 2,
    k: int = 1,
    directed: bool = False,
    weighted: bool = False,
    beta: float = 2,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a 2D Kleinberg small-world graph on an ``n x n`` toroidal grid and returns
    its adjacency matrix as a Tensor.

    Each node corresponds to a position on the 2D torus (i, j). Local edges connect each
    node to its 4 immediate neighbors (up, down, left, right) with wrapping. Additionally,
    each node gains ``k`` long-range edges, where the probability of connecting to a
    particular node depends on the toroidal Manhattan distance raised to the power ``-q``.

    When ``weighted=True``, weights are assigned as ``distance^beta`` for long-range links.

    Parameters
    ----------
    n : int
        Dimension of the grid; total nodes = n^2.
    q : float, optional
        Exponent controlling the probability of long-range connections. Default: 2.
    k : int, optional
        Number of long-range connections per node. Default: 1.
    directed : bool, optional
        If True, graph is directed; otherwise, undirected. Default: False.
    weighted : bool, optional
        If True, weight of each long-range link is ``distance^beta``; otherwise, it is
        randomly chosen from {-1, 1}. Default: False.
    beta : float, optional
        Exponent used when computing long-range weights if ``weighted=True``. Default: 2.
    seed : int or np.random.Generator or None, optional
        Seed for the RNG.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the Kleinberg small-world graph.
    """
    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    def toroidal_manhattan(i1, j1, i2, j2):
        # Wrap distances on a torus
        di = min(abs(i1 - i2), n - abs(i1 - i2))
        dj = min(abs(j1 - j2), n - abs(j1 - j2))
        return di + dj

    # Create nodes
    for i in range(n):
        for j in range(n):
            # Assign a random weight to the node (optionally used or not)
            G.add_node((i, j), weight=rng.choice([-1, 1]))

    # Local edges to 4 neighbors (toroidal wrap)
    for i in range(n):
        for j in range(n):
            neighbors = [
                ((i - 1) % n, j),  # up
                ((i + 1) % n, j),  # down
                (i, (j - 1) % n),  # left
                (i, (j + 1) % n),  # right
            ]
            for neighbor in neighbors:
                weight = rng.choice([-1, 1])
                G.add_edge((i, j), neighbor, weight=weight)
                if not directed:
                    G.add_edge(neighbor, (i, j), weight=weight)

    # Add k long-range connections per node
    for i in range(n):
        for j in range(n):
            candidates = [
                (x, y) for x in range(n) for y in range(n) if (x, y) != (i, j)
            ]
            distances = np.array(
                [toroidal_manhattan(i, j, x, y) for (x, y) in candidates], dtype=float
            )

            # Probability ~ distance^-q
            probs = distances**-q
            probs /= probs.sum()

            k_eff = min(k, len(candidates))
            chosen = rng.choice(len(candidates), size=k_eff, replace=False, p=probs)
            for idx in chosen:
                target = candidates[idx]
                dist = toroidal_manhattan(i, j, *target)
                weight = (dist**beta) if weighted else rng.choice([-1, 1])
                G.add_edge((i, j), target, weight=weight)
                if not directed:
                    G.add_edge(target, (i, j), weight=weight)

    return G


@to_tensor
def regular(
    n: int,
    k: int,
    directed: bool = False,
    self_loops: bool = False,
    random_weights: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Union[DiGraph, Graph]:
    """
    Generates a regular ring-lattice graph (each node has k neighbors) and returns
    its adjacency matrix as a Tensor.

    .. note::
        - If ``directed=True``, each undirected edge is replaced with two directed edges.
        - If ``self_loops=True``, each node also has a self-loop.
        - Weights can either be random in {-1, 1} or deterministically alternating.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Number of neighbors each node is connected to. Must be <= n - (n % 2) in undirected mode.
    directed : bool, optional
        If True, the graph is directed; else undirected. Default: False.
    self_loops : bool, optional
        If True, adds a self-loop to each node. Default: False.
    random_weights : bool, optional
        If True, weights are drawn from {-1, 1} randomly; otherwise, they alternate according
        to (-1)^(i + j). Default: True.
    seed : int or np.random.Generator or None, optional
        Seed for the RNG.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the generated regular ring-lattice graph.

    Raises
    ------
    ValueError
        If ``k > n - (n % 2)`` (an invalid regular ring-lattice configuration).
    """
    if k > n - (n % 2):
        raise ValueError(f"k must be <= n - (n % 2). Got k={k}, n={n}.")

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Connect each node to k/2 neighbors on each side
    for i in range(n):
        for j in range(1, (k // 2) + 1):
            neighbor = (i + j) % n
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** (i + neighbor)
            G.add_edge(i, neighbor, weight=weight)
            if directed:
                weight = (
                    rng.choice([-1, 1]) if random_weights else (-1) ** (i + neighbor)
                )
                G.add_edge(neighbor, i, weight=weight)

    if self_loops:
        for i in range(n):
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** i
            G.add_edge(i, i, weight=weight)

    return G


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
