import functools
import inspect
import logging
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, Union

import networkx as nx
import tensorflow as tf


def predict_factory(model: tf.keras.Model) -> Callable:
    """Create a factory function for predicting with the model.

    This function creates a factory function that can be used to predict with the model.
    The factory function is a wrapper around the model's predict method that sets the
    model's training parameter to False.

    Notes
    -----
    - The predict function is compiled with ``jit_compile=True`` for maximal
    performance on supported hardware.  Disable JIT by editing the
    decorator if XLA is not available in your environment.
    - The factory has a cache to avoid recompiling the function for the same model. This cache is of size 1 only to avoid memory issues.
    """
    if not hasattr(predict_factory, "_cache"):
        predict_factory._cache = {}

    model_id = id(model)
    if model_id in predict_factory._cache:
        return predict_factory._cache[model_id]

    # Clean up any existing cached items to prevent memory leaks
    # Only keep at most 1 item in the cache
    if len(predict_factory._cache) > 0:
        predict_factory._cache.clear()

    @tf_function(reduce_retracing=True, jit_compile=True)
    def predict(data: tf.Tensor) -> tf.Tensor:
        """Predict the output of the model.

        Args:
            data: The input data to the model.

        Returns:
            The output of the model.
        """
        return model(data, training=False)

    predict_factory._cache[model_id] = predict
    return predict


def warm_forward_factory(model: tf.keras.Model) -> Callable:
    """
    Create a factory function for warm-forwarding through the model.

    This function creates a factory function that can be used to warm-forward
    through the model. The factory function is a wrapper around the model's
    predict method that sets the model's training parameter to False.

    Notes
    -----
    - The warm_forward function is compiled with ``jit_compile=True`` for maximal
    performance on supported hardware.  Disable JIT by editing the
    decorator if XLA is not available in your environment.
    - The factory has a cache to avoid recompiling the function for the same model. This cache is of size 1 only to avoid memory issues.
    """
    if not hasattr(warm_forward_factory, "_cache"):
        warm_forward_factory._cache = {}

    model_id = id(model)
    if model_id in warm_forward_factory._cache:
        return warm_forward_factory._cache[model_id]

    # Clean up any existing cached items to prevent memory leaks
    # Only keep at most 1 item in the cache
    if len(warm_forward_factory._cache) > 0:
        warm_forward_factory._cache.clear()

    @tf_function(reduce_retracing=True, jit_compile=True)
    def warm_forward(warmup: tf.Tensor, data: tf.Tensor) -> tf.Tensor:
        """Warm-forward the model through the data.

        This function is a wrapper around the model's predict method that sets the
        model's training parameter to False.
        """
        model(warmup)  # warm-up (stateful layers adapt)
        return model(data)  # inference only - no gradients

    warm_forward_factory._cache[model_id] = warm_forward
    return warm_forward


def tf_function(*args, **kwargs) -> Callable[[Callable], Callable]:
    """Drop-in replacement for @tf.function that keeps the
    original function's __name__, __doc__, __annotations__, etc.

    Returns:
        Callable[[Callable], Callable]: A decorator that can be applied to functions to
        convert them into TensorFlow functions with the same metadata.
    """
    if args and inspect.isfunction(object=args[0]):
        # Case: used as @tf_function
        func = args[0]
        wrapped = tf.function(func=func, **kwargs)
        return functools.wraps(wrapped=func)(wrapped)
    else:
        # Case: used as @tf_function(...)
        def decorator(func: Callable) -> Callable:
            wrapped = tf.function(func=func, **kwargs)
            return functools.wraps(wrapped=func)(wrapped)
        return decorator


@contextmanager
def suppress_retracing() -> Generator[None, None, None]:
    """
    Suppress TensorFlow retracing warnings.

    This context manager temporarily suppresses TensorFlow retracing warnings
    by setting the logger level to ERROR and ignoring all warnings.

    Once the context manager is exited, the logger level is restored to its original value.

    Usage:
    >>> with suppress_retracing():
    >>>     # Your code here
    """
    prev_level = tf.get_logger().level
    tf.get_logger().setLevel(level=logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore")
        try:
            yield
        finally:
            tf.get_logger().setLevel(level=prev_level)


def suppress_retracing_during_call(fn: Callable) -> Callable:
    """
    Decorator to suppress TensorFlow retracing warnings during function calls.

    This decorator wraps the given function and ensures that TensorFlow retracing
    warnings are suppressed when the function is called. The warnings are typically
    caused by the dynamic nature of TensorFlow graphs, which can be problematic
    when running hyperparameter optimization (HPO) with Optuna.

    Parameters
    ----------
    fn : Callable
        The function to wrap.

    Returns
    -------
    Callable
        A wrapped version of the input function that suppresses TensorFlow retracing
        warnings during its execution.

    Usage
    -----
    >>> @suppress_retracing_during_call
    >>> def my_function(*args, **kwargs):
    >>>     # ...
    >>>     return ...

    Notes
    -----
    This decorator is intender to be used with functions that trigger TensorFlow retracing warnings on their call.
    """
    @wraps(wrapped=fn)
    def wrapper(*args, **kwargs) -> Any:
        with suppress_retracing():
            return fn(*args, **kwargs)
    return wrapper


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


def build_layer_graph(model: tf.keras.Model) -> nx.DiGraph:
    """
    Build a directed graph of layers from a Keras Model (Functional or subclassed).
    Works with single-/multi-input and single-/multi-output models, even when
    layers are reused or shared.

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

    # Add each layer as a node in the graph
    for layer in model.layers:
        graph.add_node(layer.name, layer_object=layer)

    # For each layer, examine all inbound nodes and their input tensors
    for layer in model.layers:
        for node in layer._inbound_nodes:
            # node.input_tensors can be a list of Tensors feeding into this node
            inbound_tensors = getattr(node, "input_tensors", [])
            for tensor in inbound_tensors:
                # _keras_history is a tuple: (layer that produced this tensor, node_index, tensor_index)
                inbound_layer = tensor._keras_history[0]
                graph.add_edge(inbound_layer.name, layer.name)

    return graph


def rebuild_model_with_new_batch_size(
    old_model: tf.keras.Model, new_batch_size: int
) -> tf.keras.Model:
    """
    Rebuild the model while modifying only the batch size of the input layers.
    Ensures that all layers (including multi-input/multi-output) are correctly
    connected in the new model. Transfers weights from the old model.

    Parameters
    ----------
    old_model : tf.keras.Model
        The original Keras model to rebuild.
    new_batch_size : int
        The desired batch size for all input layers in the new model.

    Returns
    -------
    new_model : tf.keras.Model
        A new Keras model that has the same architecture and weights as `old_model`,
        but with updated batch size in its input layers.
    """
    # 1) Build the old model's layer graph
    graph = build_layer_graph(old_model)

    # Mapping from layer name -> newly created Keras Tensor(s)
    # Note that if a layer has multiple outputs, we store them as a list/tuple.
    layer_outputs_map = {}

    # 2) Create new input layers with updated batch size
    #    Store them in `layer_outputs_map`.
    for layer_name, attrs in graph.nodes(data=True):
        layer = attrs["layer_object"]
        if isinstance(layer, tf.keras.layers.InputLayer):
            # Build a new Input with the same shape except for batch_size
            # Typically layer.batch_shape is (old_batch_size, *rest_of_shape)
            # So we skip the first dimension in `layer.batch_shape[1:]`.
            new_input = tf.keras.Input(
                shape=layer.batch_shape[1:], batch_size=new_batch_size, name=layer.name
            )
            layer_outputs_map[layer_name] = new_input

    # 3) Rebuild all non-input layers in topological order.
    #    The layer graph is a DAG, but to be safe, we do a topological sort on `graph`.
    sorted_layer_names = list(nx.topological_sort(graph))

    for layer_name in sorted_layer_names:
        # Skip if already created (e.g., InputLayers)
        if layer_name in layer_outputs_map:
            continue

        layer = graph.nodes[layer_name]["layer_object"]

        # Gather the inbound tensors for this layer by looking up the parent layers
        parent_names = list(graph.predecessors(layer_name))

        # Each parent might be single- or multi-output. The graph edges don't store which
        # output index was used, so we need to look at the *node* details to get the exact Tensors.
        # We'll do it by scanning the inbound nodes on the original layer for a match.
        inbound_tensors = []
        for node in layer._inbound_nodes:
            # For each inbound node, check if those inbound layers match parent_names
            old_inbound_tensors = getattr(node, "input_tensors", [])
            # If the inbound layers to this node match the set of parents, we use it
            old_inbound_layer_names = [
                t._keras_history[0].name for t in old_inbound_tensors
            ]
            # We do a quick check if these parents match exactly the parents we see in the graph
            # (or if it's a subset for multi-input).
            if set(old_inbound_layer_names) == set(parent_names):
                # Use *this* node's input tensors
                for t in old_inbound_tensors:
                    parent_layer_name = t._keras_history[0].name
                    parent_output = layer_outputs_map[parent_layer_name]

                    # If the parent layer had multiple outputs, we must pick the correct output index
                    output_index = t._keras_history[2]  # which output from parent layer
                    if isinstance(parent_output, (list, tuple)):
                        inbound_tensors.append(parent_output[output_index])
                    else:
                        # Single output from parent
                        inbound_tensors.append(parent_output)
                break  # we found the correct inbound node, no need to keep searching

        # For convenience, if there's exactly one inbound tensor, pass it directly
        if len(inbound_tensors) == 1:
            inbound_tensors = inbound_tensors[0]

        # 4) Recreate the layer from its config
        #    - from_config() returns an un-built layer object
        #    - We then call it on the inbound_tensors to obtain the new output Tensors
        layer_config = layer.get_config()

        # If you need to forcibly preserve name, you can do:
        # layer_config['name'] = layer.name

        new_layer = layer.__class__.from_config(layer_config)

        # Actually call it on inbound_tensors to create the new output Tensors
        # This could be a single Tensor or a list/tuple of Tensors
        new_outputs = new_layer(inbound_tensors)

        # Store these new outputs in the mapping. Could be one or many.
        # We'll always store a list if it's multiple.
        layer_outputs_map[layer_name] = new_outputs

    # 5) Identify the new model's outputs by mapping each old output Tensor
    #    to the correct new output Tensor.
    new_model_outputs = []
    for old_out, out_name in zip(old_model.outputs, old_model.output_names):
        parent_layer_of_output = old_out._keras_history[
            0
        ]  # the layer that created this tensor
        parent_layer_name = parent_layer_of_output.name
        parent_output_index = old_out._keras_history[
            2
        ]  # which output index from that layer

        new_parent_outputs = layer_outputs_map[parent_layer_name]
        if isinstance(new_parent_outputs, (list, tuple)):
            new_model_outputs.append(new_parent_outputs[parent_output_index])
        else:
            new_model_outputs.append(new_parent_outputs)

    # 6) Collect the new inputs in the correct order. The first N items in layer_outputs_map
    #    that are InputLayers might not necessarily match the original order. So let's track them
    #    by referencing old_model.inputs.
    new_model_inputs = []
    for old_in in old_model.inputs:
        in_layer = old_in._keras_history[0]  # InputLayer
        new_in = layer_outputs_map[in_layer.name]
        new_model_inputs.append(new_in)

    # If there's exactly one input, Keras expects a single tensor, not a list
    if len(new_model_inputs) == 1:
        new_model_inputs = new_model_inputs[0]
    if len(new_model_outputs) == 1:
        new_model_outputs = new_model_outputs[0]

    # 7) Construct the new model
    new_model = tf.keras.Model(inputs=new_model_inputs, outputs=new_model_outputs)

    # 8) Transfer weights
    new_model.set_weights(old_model.get_weights())

    return new_model


def insert_layer(
    model: tf.keras.Model, after_layer: str, new_layer: tf.keras.layers.Layer
) -> tf.keras.Model:
    """
    Insert `new_layer` immediately after the layer named `after_layer` in the
    functional graph of `model`. The new layer is assumed to have shape-compatible
    input -> output so that it doesn't break downstream layers.

    Parameters
    ----------
    model : tf.keras.Model
        The source model (Functional).
    after_layer : str
        The name of the layer after which `new_layer` will be inserted.
    new_layer : keras.layers.Layer
        The new layer instance to insert. This layer must be "callable"
        (i.e. not yet called on a Tensor) when passed in, or else from_config logic
        can get complicated.

    Returns
    -------
    new_model : tf.keras.Model
        A new model with `new_layer` inserted after `after_layer`.
    """
    # 1) Build the old model's layer graph
    graph = build_layer_graph(model)

    if after_layer not in graph:
        raise ValueError(f"Layer '{after_layer}' not found in model.")

    # 2) Insert a new node for the new layer in the graph
    #    We'll store it as though it's another layer in the model.
    new_layer_name = new_layer.name
    if new_layer_name in graph:
        raise ValueError(
            f"The new layer's name '{new_layer_name}' already exists in the model. "
            "Please assign a unique name."
        )

    graph.add_node(new_layer_name, layer_object=new_layer)

    # 3) Rewire edges: any edge (after_layer -> X) becomes (after_layer -> new_layer -> X)
    successors = list(graph.successors(after_layer))
    # Remove old edges from `after_layer` to its successors
    for succ in successors:
        graph.remove_edge(after_layer, succ)
    # Add edge from `after_layer` to `new_layer`
    graph.add_edge(after_layer, new_layer_name)
    # Then add edges from `new_layer` to each successor
    for succ in successors:
        graph.add_edge(new_layer_name, succ)

    # 4) Rebuild the model from the updated graph
    #    We'll do something similar to "rebuild_model_with_new_batch_size", but
    #    we adapt it for the insertion. The main difference is that we skip from_config
    #    for the newly inserted layer and directly use the user-supplied `new_layer`.

    # Step A: topological sort
    sorted_layer_names = list(nx.topological_sort(graph))

    # Mapping from layer name -> newly created Keras Tensor(s)
    layer_outputs_map = {}

    # B. Recreate Input layers first (unchanged)
    for layer_name in sorted_layer_names:
        layer_obj = graph.nodes[layer_name]["layer_object"]
        if isinstance(layer_obj, tf.keras.layers.InputLayer):

            batch_shape = layer_obj.batch_shape

            # Create new Input
            new_input = tf.keras.Input(
                shape=batch_shape[1:],
                batch_size=batch_shape[0],
                name=layer_obj.name,
            )
            layer_outputs_map[layer_name] = new_input

    # C. Recreate all other layers in topological order
    for layer_name in sorted_layer_names:
        # If it's already created (InputLayer or something else), skip
        if layer_name in layer_outputs_map:
            continue

        layer_obj = graph.nodes[layer_name]["layer_object"]
        parent_names = list(graph.predecessors(layer_name))

        # Gather inbound tensors
        inbound_tensors = []
        # We find the original inbound node in the old model that matches these parents (if it exists).
        # But because we've possibly changed the graph (inserting a new layer), we must rely on
        # the edges and the parent's output mapping, plus the original _keras_history indexing if possible.
        # We'll do a simpler approach: we know each parent might have single or multiple outputs,
        # so let's see how we can gather them.

        # We scan old inbound nodes to guess the correct set, but if this is the newly inserted layer,
        # we won't find it in the old model. So for that, we rely purely on the parent's new outputs.
        if layer_name == new_layer_name:
            # The newly inserted layer has exactly the parents we rewired from `after_layer`.
            # So we gather the new Tensors from `after_layer` (which might be single or multi-output).
            # Possibly multiple parent_names if after_layer had multiple outputs or if it's multi-output.
            # We'll collect them all.
            for pn in parent_names:
                parent_out = layer_outputs_map[pn]
                if isinstance(parent_out, (list, tuple)):
                    # If the parent had multiple outputs, we feed them all in if the new layer is multi-input
                    inbound_tensors.extend(parent_out)
                else:
                    inbound_tensors.append(parent_out)

            # Now we directly call the user-supplied new_layer object to produce output
            if len(inbound_tensors) == 1:
                inbound_tensors = inbound_tensors[0]
            new_outputs = layer_obj(inbound_tensors)  # call the actual new_layer
            layer_outputs_map[layer_name] = new_outputs
            continue

        # Otherwise, this is a pre-existing layer from the old model. We do the usual approach.
        # We look at all inbound nodes in the old layer to find which one matches parent_names.
        old_inbound_nodes = getattr(layer_obj, "_inbound_nodes", [])
        found = False
        for node in old_inbound_nodes:
            old_inbound_tensors = getattr(node, "input_tensors", [])
            old_inbound_parents = [
                t._keras_history[0].name for t in old_inbound_tensors
            ]
            # If the set or list order matches, let's consider it a match
            # (some models might reorder, so we do a simple set check).
            if set(old_inbound_parents) == set(parent_names):
                # We use these inbound_tensors
                for t in old_inbound_tensors:
                    parent_layer_name = t._keras_history[0].name
                    parent_output = layer_outputs_map[parent_layer_name]
                    output_index = t._keras_history[2]  # index of the parent output
                    if isinstance(parent_output, (list, tuple)):
                        inbound_tensors.append(parent_output[output_index])
                    else:
                        inbound_tensors.append(parent_output)
                found = True
                break

        if not found:
            # If we never found a matching node in the old model,
            # we rely on the parent's new Tensors from the graph edges.
            for pn in parent_names:
                parent_out = layer_outputs_map[pn]
                if isinstance(parent_out, (list, tuple)):
                    inbound_tensors.extend(parent_out)
                else:
                    inbound_tensors.append(parent_out)

        if len(inbound_tensors) == 1:
            inbound_tensors = inbound_tensors[0]

        # Recreate the layer from config
        layer_config = layer_obj.get_config()
        # We can preserve the original name or let from_config handle it
        # layer_config["name"] = layer_obj.name

        cloned_layer = layer_obj.__class__.from_config(layer_config)
        # Call it
        new_outputs = cloned_layer(inbound_tensors)
        layer_outputs_map[layer_name] = new_outputs

    # D. Identify the new model's outputs by referencing old_model.outputs
    new_model_outputs = []
    for old_out, out_name in zip(model.outputs, model.output_names):
        parent_layer_of_output = old_out._keras_history[0]
        parent_output_index = old_out._keras_history[2]

        new_parent_outputs = layer_outputs_map[parent_layer_of_output.name]
        if isinstance(new_parent_outputs, (list, tuple)):
            new_model_outputs.append(new_parent_outputs[parent_output_index])
        else:
            new_model_outputs.append(new_parent_outputs)

    # E. Collect the new inputs in the correct order
    new_model_inputs = []
    for old_in in model.inputs:
        in_layer = old_in._keras_history[0]  # InputLayer
        new_in = layer_outputs_map[in_layer.name]
        new_model_inputs.append(new_in)

    if len(new_model_inputs) == 1:
        new_model_inputs = new_model_inputs[0]
    if len(new_model_outputs) == 1:
        new_model_outputs = new_model_outputs[0]

    # F. Construct the new model
    new_model = tf.keras.Model(inputs=new_model_inputs, outputs=new_model_outputs)

    # G. Transfer weights, skipping the newly inserted layer
    for layer in new_model.layers:
        if layer.name == new_layer_name:     # freshly inserted layer
            continue
        old_layer = model.get_layer(layer.name)
        layer.set_weights(old_layer.get_weights())

    return new_model


__all__ = [
    "build_layer_graph",
    "create_tf_rng",
    "insert_layer",
    "predict_factory",
    "rebuild_model_with_new_batch_size",
    "suppress_retracing",
    "suppress_retracing_during_call",
    "tf_function",
    "warm_forward_factory",
]


def __dir__():
    return __all__
