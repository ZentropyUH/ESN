from typing import Optional, Union

import keras
import networkx as nx
import tensorflow as tf
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


def build_layer_graph(model: Model) -> nx.DiGraph:
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

    # Add all layers as nodes
    for layer in model.layers:
        graph.add_node(layer.name, layer=layer)

    # Add edges to represent connections between layers
    for layer in model.layers:
        inbound_layers = []
        if isinstance(layer.input, (list, tuple)):
            for inp in layer.input:
                if hasattr(inp, "_keras_history"):
                    inbound_layers.append(inp._keras_history[0])
        elif hasattr(layer.input, "_keras_history"):
            inbound_layers.append(layer.input._keras_history[0])

        for inbound_layer in inbound_layers:
            graph.add_edge(inbound_layer.name, layer.name)

    return graph


def rebuild_model_with_new_batch_size(old_model: Model, new_batch_size: int) -> Model:
    """
    Rebuild the model while modifying only the batch size of the input layers.
    Ensures that all layers are correctly connected in the new model.
    """
    graph = build_layer_graph(old_model)
    layer_mapping = {}

    # Create new input layers with updated batch size
    for layer_name, attrs in graph.nodes(data=True):
        layer = attrs["layer"]
        if isinstance(layer, keras.layers.InputLayer):
            new_input = keras.Input(
                shape=layer.batch_shape[1:], batch_size=new_batch_size, name=layer.name
            )
            layer_mapping[layer_name] = new_input

    # Rebuild all other layers while maintaining their original connections
    for layer_name, attrs in graph.nodes(data=True):
        if layer_name in layer_mapping:  # Skip input layers (already handled)
            continue

        layer = attrs["layer"]
        inbound_nodes = list(graph.predecessors(layer_name))  # Get parent layers
        inbound_tensors = [layer_mapping[parent] for parent in inbound_nodes]

        # Ensure single inputs are not enclosed in a list
        if len(inbound_tensors) == 1:
            inbound_tensors = inbound_tensors[0]

        # Recreate the layer with the same config, connected to the proper inputs
        new_layer = layer.__class__.from_config(layer.get_config())(inbound_tensors)
        layer_mapping[layer_name] = new_layer

    # Extract the final outputs
    new_outputs = [
        layer_mapping[old_model.output_names[i]] for i in range(len(old_model.outputs))
    ]

    if len(new_outputs) == 1:
        new_outputs = new_outputs[0]

    inputs = list(layer_mapping.values())[: len(old_model.inputs)]
    if len(inputs) == 1:
        inputs = inputs[0]

    # Create the new model
    new_model = Model(
        inputs=inputs,
        outputs=new_outputs,
    )

    # Load the trained weights
    new_model.set_weights(old_model.get_weights())

    return new_model


__all__ = [
    "create_tf_rng",
    "build_layer_graph",
]


def __dir__():
    return __all__
