from networkx import DiGraph
import tensorflow as tf
from typing import Optional, Union
import networkx as nx
from keras import Model


def create_tf_rng(
    seed: Optional[Union[int, tf.random.Generator]] = None
) -> tf.random.Generator:
    """
    Create and return a TensorFlow random number generator (RNG).

    Parameters
    ----------
    seed : int, tf.random.Generator, or None, optional
        - If an integer, a new deterministic generator is created using that seed.
        - If a tf.random.Generator, it is returned as is.
        - If None, a new generator is created from a non-deterministic state.

    Returns
    -------
    tf.random.Generator
        A TensorFlow random number generator instance.

    Raises
    ------
    TypeError
        If `seed` is neither an integer, a `tf.random.Generator`, nor None.

    Examples
    --------
    >>> # Creating a deterministic generator:
    >>> rng = create_tf_rng(seed=42)
    >>> sample = rng.normal(shape=(2, 2))

    >>> # Creating a non-deterministic generator:
    >>> rng2 = create_tf_rng()
    """
    # Decide how to create or return the generator
    if isinstance(seed, int):
        rg = tf.random.Generator.from_seed(seed)
    elif isinstance(seed, tf.random.Generator):
        rg = seed
    elif seed is None:
        rg = tf.random.Generator.from_non_deterministic_state()
    else:
        raise TypeError("`seed` must be an integer, tf.random.Generator, or None.")

    return rg


def build_layer_graph(model: Model) -> DiGraph:
    """
    Build a directed graph of layers from a Functional Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model from which to build the layer graph.

    Returns
    -------
    graph : networkx.DiGraph
        A directed graph where nodes are layer names and edges represent the
        connections between layers.
    """
    graph = nx.DiGraph()

    # Add all layers as graph nodes
    for layer in model.layers:
        graph.add_node(layer.name, layer=layer)

    # Build edges: parent -> child
    for layer in model.layers:
        inbound_layers = []

        if isinstance(layer.input, (list, tuple)):  # Ensure compatibility with tuples
            for inp in layer.input:
                if hasattr(inp, "_keras_history"):
                    inbound_layers.append(
                        inp._keras_history[0]
                    )  # The actual parent layer
        elif hasattr(layer.input, "_keras_history"):  # Handle single tensor case
            inbound_layers.append(layer.input._keras_history[0])

        # Add edges from each inbound layer to the current layer
        for inbound_layer in inbound_layers:
            graph.add_edge(inbound_layer.name, layer.name)

    return graph


__all__ = [
    "create_tf_rng",
    "build_layer_graph",
]


def __dir__():
    return __all__
