from typing import Union
import numpy as np
from  networkx import DiGraph, Graph
from .helpers import create_rng, to_tensor, connected_graph


@to_tensor
def watts_strogatz(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: Union[int, np.random.Generator, None] = None,
) -> Union[Graph, DiGraph]:
    """
    Generates a Watts-Strogatz small-world graph with rewiring probability p.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to k/2 predecessors and k/2 successors.
    p : float
        Rewiring probability.
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected one.
    self_loops : bool, optional
        If True, allows self-loops during rewiring.
    seed : int, np.random.Generator, or None, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The generated Watts-Strogatz graph.
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n})")

    if k % 2 != 0:
        k += 1  # Ensure k is even

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initial ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n, weight=rng.choice([-1, 1]))  # Forward
            if directed:
                G.add_edge(i, (i - j) % n, weight=rng.choice([-1, 1]))  # Backward

    edges = list(G.edges())  # Ensure a stable edge list

    # Rewire edges with probability p
    for u, v in edges:
        if rng.random() < p:
            G.remove_edge(u, v)  # Watts-Strogatz removes the original edge

            candidates = rng.permutation(n)  # Shuffle node choices
            for new_v in candidates:
                if (new_v != u or self_loops) and not G.has_edge(u, new_v):
                    G.add_edge(u, new_v, weight=rng.choice([-1, 1]))
                    break  # Found a valid new_v, stop searching

    return G


@connected_graph
def connected_watts_strogatz(
    n: int,
    k: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: Union[int, np.random.Generator, None] = None,
) -> Union[Graph, DiGraph]:
    """
    Generates a connected Watts-Strogatz graph with rewiring probability p.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to k/2 predecessors and k/2 successors.
    p : float
        Rewiring probability.
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected one. Default is True.
    self_loops : bool, optional
        If True, allows self-loops during rewiring. Default is True.
    seed : int or None, optional
        Seed for the random number generator. Default is None.
    tries : int, optional
        Number of attempts to generate a connected graph. **Handled by the `connected_graph` decorator**. Default is 100.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the connected Watts-Strogatz graph.
    """
    return watts_strogatz(n, k, p, directed, self_loops, seed)


@to_tensor
def erdos_renyi(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: Union[int, np.random.Generator, None] = None,
) -> Union[Graph, DiGraph]:
    """
    Generates an Erdos-Renyi graph with edge probability p.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability.
    directed : bool
        If True, generates a directed graph; otherwise, an undirected one.
    self_loops : bool
        If True, allows self-loops in the graph.
    seed : int, np.random.Generator, or None
        Seed for the random number generator. Default is None.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Erdos-Renyi graph.
    """
    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    nodes = range(n)
    G.add_nodes_from(nodes)  # Ensure all nodes are included

    if directed:
        edges = [(u, v) for u in nodes for v in nodes if self_loops or u != v]
    else:
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
    seed: Union[int, np.random.Generator, None] = None,
) -> Union[Graph, DiGraph]:
    """
    Generates a connected Erdos-Renyi graph with edge probability p.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability.
    directed : bool
        If True, generates a directed graph; otherwise, an undirected one.
    self_loops : bool
        If True, allows self-loops in the graph.
    seed : int, np.random.Generator, or None
        Seed for the random number generator. Default is None.
    tries : int, optional
        Number of attempts to generate a connected graph. **Handled by the `connected_graph` decorator**. Default is 100.

    Returns
    -------
    tf.Tensor
        Adjacency matrix of the connected Erdos-Renyi graph.
    """
    return erdos_renyi(n, p, directed, self_loops, seed)


@to_tensor  # Newman-Watts-Strogatz is always connected
def newman_watts_strogatz(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: Union[int, np.random.Generator, None] = None,
) -> Union[Graph, DiGraph]:
    """
    Generates a Newman-Watts-Strogatz small-world graph with rewiring probability p.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to k/2 predecessors and k/2 successors.
    p : float
        Rewiring probability.
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected one.
    self_loops : bool, optional
        If True, allows self-loops during rewiring.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The generated Newman-Watts-Strogatz graph.
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n})")

    if k % 2 != 0:
        k += 1  # Ensure k is even

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initial ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n, weight=rng.choice([-1, 1]))  # Forward
            if directed:
                G.add_edge(i, (i - j) % n, weight=rng.choice([-1, 1]))  # Backward

    edges = list(G.edges())  # Ensure a stable edge list

    # Rewire edges with probability p (without removing existing edges)
    for u, v in edges:
        if rng.random() < p:
            candidates = rng.permutation(n)  # Shuffle node choices
            for new_v in candidates:
                if (new_v != u or self_loops) and not G.has_edge(u, new_v):
                    G.add_edge(u, new_v, weight=rng.choice([-1, 1]))
                    break  # Found a valid new_v, stop searching

    return G


@to_tensor
def barabasi_albert(
    n: int,
    m: int,
    directed: bool = False,
    seed: Union[int, np.random.Generator, None] = None,
) -> Union[Graph, DiGraph]:
    if m < 1 or m >= n:
        raise ValueError(f"m must be >= 1 and < n, got m={m}, n={n}.")

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Start with a fully connected core of m nodes
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            weight = rng.choice([-1, 1])
            G.add_edge(i, j, weight=weight)
            if directed:
                G.add_edge(j, i, weight=rng.choice([-1, 1]))

    # List to store nodes with degree, allowing for faster attachment
    targets = list(G.nodes) * m

    for i in range(m, n):
        G.add_node(i)
        new_edges = rng.choice(targets, size=m, replace=False)
        for t in new_edges:
            weight = rng.choice([-1, 1])
            G.add_edge(i, t, weight=weight)
            if directed:
                G.add_edge(t, i, weight=rng.choice([-1, 1]))

        targets.extend([i] * m)
        targets.extend(new_edges)

    return G


@to_tensor
def kleinberg_small_world(
    n: int,
    q: int = 2,
    k: int = 1,
    directed: bool = False,
    weighted: bool = False,
    beta: float = 2,
    seed: Union[int, np.random.Generator, None] = None,
):
    """
    Generates a 2D Kleinberg small-world graph on an n x n toroidal grid.
    Total number of nodes = n^2.

    Each node (i, j) has 4 nearest neighbors in a wrapping/toroidal fashion:
        - up:    (i-1) mod n, j
        - down:  (i+1) mod n, j
        - left:  i, (j-1) mod n
        - right: i, (j+1) mod n
    Then each node is assigned 'k' long-range links with probability
    P(d) ∝ d^-q, where d is the toroidal Manhattan distance.

    Parameters
    ----------
    n : int
        Grid dimension. Total nodes = n^2.
    q : float, optional
        Exponent controlling long-range connection probability decay (default: 2).
    k : int, optional
        Number of long-range connections per node (default: 1).
    directed : bool, optional
        If True, generates a directed graph; otherwise, undirected (default: False).
    weighted : bool, optional
        If True, assigns weights proportional to distance^beta (default: False).
    beta : float, optional
        Exponent for weight calculation when weighted=True (default: 2).
    seed : int, np.random.Generator, or None
        Seed for random number generator.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Kleinberg small-world graph on an n x n torus.
    """
    rng = np.random.default_rng(seed)
    G = DiGraph() if directed else Graph()

    # Helper function: Toroidal Manhattan distance
    def toroidal_manhattan(i1, j1, i2, j2):
        di = min(abs(i1 - i2), n - abs(i1 - i2))  # Wrap vertically
        dj = min(abs(j1 - j2), n - abs(j1 - j2))  # Wrap horizontally
        return di + dj

    # Step 1: Add n^2 nodes labeled (i, j)
    for i in range(n):
        for j in range(n):
            G.add_node((i, j), weight=rng.choice([-1, 1]))

    # Step 2: Connect each node to its 4 nearest neighbors (toroidal)
    for i in range(n):
        for j in range(n):
            neighbors = [
                ((i - 1) % n, j),  # Up
                ((i + 1) % n, j),  # Down
                (i, (j - 1) % n),  # Left
                (i, (j + 1) % n),  # Right
            ]
            for neighbor in neighbors:
                weight = rng.choice([-1, 1])
                G.add_edge((i, j), neighbor, weight=weight)
                if not directed:
                    G.add_edge(
                        neighbor, (i, j), weight=weight
                    )  # Ensure bidirectionality

    # Step 3: Add k long-range connections per node
    for i in range(n):
        for j in range(n):
            candidates = [
                (x, y) for x in range(n) for y in range(n) if (x, y) != (i, j)
            ]

            distances = np.array(
                [toroidal_manhattan(i, j, x, y) for (x, y) in candidates], dtype=float
            )

            # Probability distribution ~ d^-q
            probs = distances**-q
            probs /= probs.sum()

            n_candidates = len(candidates)
            k = min(k, n_candidates)

            chosen_indices = rng.choice(n_candidates, size=k, replace=False, p=probs)
            for idx in chosen_indices:
                target = candidates[idx]
                distance = toroidal_manhattan(i, j, *target)
                weight = distance ** beta if weighted else rng.choice([-1, 1])
                G.add_edge((i, j), target, weight=weight)
                if not directed:
                    G.add_edge(target, (i, j), weight=weight)

    return G
