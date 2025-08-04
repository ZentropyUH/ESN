"""
State management utilities for reservoir layers in TensorFlow/Keras models.
This module provides functions for inspecting, manipulating, and analyzing the internal
states of reservoir computing layers in Keras models. It includes functionality to get
and set reservoir states, reset states, harvest states during sequence processing,
and calculate metrics like the Echo State Property index.
Functions
---------
get_reservoir_states(model)
set_reservoir_states(model, states)
set_reservoir_random_states(model, dist='uniform')
harvest(model, feedback_seq, external_seqs=())
esp_index(model, feedback_seq, external_seqs=(), random_dist='uniform',
          history=False, weighted=False, iterations=10)
This module is designed to work with the keras_reservoir_computing package and
specifically with keras models that contain layers that inherit from BaseReservoir.
"""

from typing import Optional, Tuple, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.layers.base import BaseReservoir
from keras_reservoir_computing.utils.tensorflow import tf_function, create_tf_rng


def get_reservoir_states(model: tf.keras.Model) -> dict:
    """
    Get the current states of all reservoir layers in the model.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing the reservoir layers.

    Returns
    -------
    dict
        A dictionary mapping reservoir layer names to their current states.
        Each value is a list of tensors, with shape [batch_size, state_size].

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.io.loaders import load_object, load_default_config
    >>> from keras_reservoir_computing.analysis.states import get_reservoir_states
    >>>
    >>> inputs = tf.keras.Input(shape=(None, 5), batch_size=2)
    >>> reservoir_cfg = load_default_config("reservoir")
    >>> reservoir = load_object(reservoir_cfg)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> states = get_reservoir_states(model)
    >>> for layer_name, layer_states in states.items():
    ...     print(f"Layer {layer_name} states: {[s.shape for s in layer_states]}")
    """
    states = {}

    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            states[layer.name] = layer.get_states()

    return states


def set_reservoir_states(model: tf.keras.Model, states: dict) -> None:
    """
    Set the states of all reservoir layers in the model.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing the reservoir layers.
    states : dict
        A dictionary mapping reservoir layer names to their desired states.
        Each value should be a list of tensors with shape [batch_size, state_size],
        matching the state structure of each reservoir layer.

    Raises
    ------
    ValueError
        If the keys in the states dictionary do not match the names of the reservoir layers
        in the model, or if state dimensions don't match the reservoir configuration.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.io.loaders import load_object, load_default_config
    >>> from keras_reservoir_computing.analysis.states import get_reservoir_states, set_reservoir_states
    >>>
    >>> inputs = tf.keras.Input(shape=(None, 5), batch_size=2)
    >>> reservoir_cfg = load_default_config("reservoir")
    >>> reservoir = load_object(reservoir_cfg)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> current_states = get_reservoir_states(model)
    >>> custom_states = {
    ...     name: [tf.random.uniform(tf.shape(s)) for s in states]
    ...     for name, states in current_states.items()
    ... }
    >>> set_reservoir_states(model, custom_states)
    """

    # Check if the states are valid
    layer_names = [
        layer.name
        for layer in model.layers
        if isinstance(layer, BaseReservoir)
    ]

    if len(layer_names) == 0:
        return

    if set(states.keys()) != set(layer_names):
        raise ValueError(
            f"Invalid states. Expected keys in new states: {layer_names}. Got keys: {states.keys()}"
        )

    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            layer.set_states(states[layer.name])


def reset_reservoir_states(model: tf.keras.Model) -> None:
    """
    Reset the states of all reservoir layers in the model to zero.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing the reservoir layers.

    Returns
    -------
    None
        The method updates the reservoir states in-place.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.io.loaders import load_object, load_default_config
    >>> from keras_reservoir_computing.analysis.states import reset_reservoir_states
    >>>
    >>> inputs = tf.keras.Input(shape=(None, 5), batch_size=2)
    >>> reservoir_cfg = load_default_config("reservoir")
    >>> reservoir = load_object(reservoir_cfg)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> reset_reservoir_states(model)
    """

    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            layer.reset_states()


def set_reservoir_random_states(model: tf.keras.Model, dist: str = "uniform", seed: Optional[int] = None) -> None:
    """
    Set the states of all reservoir layers in the model to random values.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing the reservoir layers.
    dist : str, optional
        Distribution to use for random state initialization.
        Options are "uniform" (default) or "normal".
    seed : Optional[int], optional
        Random seed for reproducibility.

    Returns
    -------
    None
        The method updates the reservoir states in-place.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.io.loaders import load_object, load_default_config
    >>> from keras_reservoir_computing.analysis.states import set_reservoir_random_states
    >>>
    >>> inputs = tf.keras.Input(shape=(None, 5), batch_size=2)
    >>> reservoir_cfg = load_default_config("reservoir")
    >>> reservoir = load_object(reservoir_cfg)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> set_reservoir_random_states(model, dist="uniform")
    >>> set_reservoir_random_states(model, dist="normal")
    """
    # Create a single random generator to avoid creating many separate generators
    rng = create_tf_rng(seed)

    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            layer.set_random_states(dist=dist, seed=rng)



@tf_function
def harvest(
    model: tf.keras.Model,
    feedback_seq: tf.Tensor,
    external_seqs: Tuple[tf.Tensor, ...] = (),
) -> dict:
    """
    Collect reservoir states while running the model with provided sequences.

    This function runs the model step by step with the given input sequences
    and collects the internal states of all reservoir layers at each timestep.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing BaseReservoir layers whose states will be collected.
    feedback_seq : tf.Tensor
        The feedback sequence to use, shape [batch_size, timesteps, features].
    external_seqs : Tuple[tf.Tensor, ...], optional
        Tuple of external input sequences, each shaped [batch_size, timesteps, features].
        Should be in the same order as the model's additional inputs. Default is an empty tuple.

    Returns
    -------
    dict
        Dictionary mapping layer names to lists of state tensors.
        Each state tensor has shape [batch_size, timesteps, state_size].

    Notes
    -----
    - The function uses a TensorFlow while loop to iterate over the sequences.
    - States are collected using TensorArrays and stacked at the end.
    - The function is graph-mode compatible (@tf.function decorated).
    - A ControlTrigger is used to prevent race conditions when updating states.
    - The minimum timestep dimension from all inputs is used as the horizon.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.io.loaders import load_object, load_default_config
    >>> from keras_reservoir_computing.analysis.states import harvest
    >>>
    >>> inputs = tf.keras.Input(shape=(None, 1), batch_size=2)
    >>> reservoir_cfg = load_default_config("reservoir")
    >>> reservoir = load_object(reservoir_cfg)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> feedback_seq = tf.random.normal((2, 10, 1))
    >>> states = harvest(model, feedback_seq)
    >>> for layer_name, layer_states in states.items():
    ...     print(f"Layer {layer_name} states: {[s.shape for s in layer_states]}")
    Layer esn_reservoir_1 states: [(2, 10, 100)]  # [batch_size, timesteps, units]
    """
    input_names = [_input.name for _input in model.inputs]
    if len(input_names) < 1:
        raise ValueError("Model must have at least one input (the feedback input).")

    horizon = tf.reduce_min(
        [tf.shape(feedback_seq)[1]] + [tf.shape(v)[1] for v in external_seqs]
    )

    external_input_count = len(input_names) - 1
    if len(external_seqs) != external_input_count:
        raise ValueError(
            f"Expected {external_input_count} external inputs, but got {len(external_seqs)}."
        )

    t0 = tf.constant(0, dtype=tf.int32)

    states = {}
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            states[layer.name] = [
                tf.TensorArray(dtype=state.dtype, size=horizon, infer_shape=True)
                for state in layer.get_states()
            ]

    loop_vars = (t0, states)

    def cond(t, states):
        return t < horizon

    def body(t, states):
        model_inputs = [tf.expand_dims(feedback_seq[:, t, :], axis=1)]
        if external_seqs:
            model_inputs.extend(
                [tf.expand_dims(ext[:, t, :], axis=1) for ext in external_seqs]
            )

        model_inputs = model_inputs[0] if len(model_inputs) == 1 else model_inputs

        _ = model(model_inputs)

        # Force execution order to avoid race conditions
        tf.raw_ops.ControlTrigger()

        for layer in model.layers:
            if isinstance(layer, BaseReservoir):
                for i, (st_ta, st) in enumerate(
                    zip(states[layer.name], layer.get_states())
                ):
                    states[layer.name][i] = st_ta.write(t, st)

        return t + 1, states

    shape_invariants = (
        t0.shape,
        {
            layer.name: [tf.TensorShape(None) for _ in layer.get_states()]
            for layer in model.layers
            if isinstance(layer, BaseReservoir)
        },
    )

    _, states_history = tf.while_loop(
        cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants
    )

    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            for i, st_ta in enumerate(states_history[layer.name]):
                states_history[layer.name][i] = tf.transpose(st_ta.stack(), [1, 0, 2])

    return states_history


def esp_index(
    model: tf.keras.Model,
    feedback_seq: tf.Tensor,
    external_seqs: Tuple[tf.Tensor, ...] = (),
    random_dist: str = "uniform",
    history: bool = False,
    iterations: int = 10,
    transient: int = 0,
) -> Union[dict, Tuple[dict, dict]]:
    """
    Compute the Echo State Property (ESP) index for reservoir layers in the model.

    The ESP index quantifies how quickly the reservoir forgets its initial state,
    which is a critical property of Echo State Networks. The index is computed by
    measuring how quickly trajectories from different initial states converge.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing BaseReservoir layers whose ESP index will be computed.
    feedback_seq : tf.Tensor
        The feedback sequence to use, shape [batch_size, timesteps, features].
    external_seqs : Tuple[tf.Tensor, ...], optional
        Tuple of external input sequences, each with shape [batch_size, timesteps, features].
        Should be in the same order as the model's additional inputs. Default is an empty tuple.
    random_dist : str, optional
        Distribution to use for random state initialization: "uniform" or "normal".
        Default is "uniform".
    history : bool, optional
        Whether to return the full time evolution of state difference norms.
        Default is False.
    iterations : int, optional
        Number of iterations to use for computing the ESP index. Higher values
        give more reliable estimates. Default is 10.
    transient: int, optional
        Number of timesteps to discard from the beginning of the sequence.
        Default is 0.
    Returns
    -------
    Union[dict, Tuple[dict, dict]]
        If history=False (default):
            dict: A dictionary mapping layer names to lists of ESP index values
                 for each state in the layer.
        If history=True:
            Tuple[dict, dict]: A tuple containing:
                - dict: The ESP indices as described above
                - dict: The full history of state difference norms for each layer and state
                  with shape [iterations, batch_size, timesteps]

    Notes
    -----
    - The ESP index calculation uses the method from Gallicchio (2019).
    - Lower ESP index values indicate better echo state property.
    - The computation involves multiple steps:
        1. Reset states and run a "base orbit" from zero initial state
        2. Run multiple "random orbits" from random initial states
        3. Measure how quickly the random orbits converge to the base orbit
        4. Average the convergence rates across iterations
    - Progress is printed to show computation status.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.io.loaders import load_object, load_default_config
    >>> from keras_reservoir_computing.analysis.states import esp_index
    >>>
    >>> inputs = tf.keras.Input(shape=(None, 1), batch_size=2)
    >>> reservoir_cfg = load_default_config("reservoir")
    >>> reservoir = load_object(reservoir_cfg)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> feedback_seq = tf.random.normal((2, 50, 1))
    >>> indices = esp_index(model, feedback_seq, iterations=5)
    >>> print(f"ESP indices: {indices}")
    >>> indices, history = esp_index(model, feedback_seq, iterations=5, history=True)
    >>> print(f"History shapes: {[h.shape for h in history[list(history.keys())[0]]]}")

    References
    ----------
    .. [1] C. Gallicchio, "Chasing the Echo State Property," Sep. 24, 2019,
       arXiv: arXiv:1811.10892. Accessed: May 30, 2023. [Online].
       Available: http://arxiv.org/abs/1811.10892
    """


    input_dtype = feedback_seq.dtype

    # --- Save current states for restoring later ---
    current_states = {
        name: [tf.identity(state) for state in states]
        for name, states in get_reservoir_states(model).items()
    }

    # --- Base orbit from zero states ---
    reset_reservoir_states(model)
    base_orbit = harvest(model, feedback_seq, external_seqs)

    # --- Initialize storage for ESP indices ---
    esp_indices = {
        layer_name: [tf.constant(0, dtype=input_dtype) for _ in states]
        for layer_name, states in base_orbit.items()
    }

    # --- Initialize storage for history if requested ---
    esp_history = None
    if history:
        esp_history = {
            layer_name: [
                tf.TensorArray(dtype=input_dtype, size=iterations)
                for _ in states
            ]
            for layer_name, states in base_orbit.items()
        }

    # --- Main loop over iterations ---
    for iter_idx in range(iterations):
        print(f"\rIteration {iter_idx + 1}/{iterations}", end="", flush=True)

        # Generate random initial states and compute orbit
        set_reservoir_random_states(model, dist=random_dist)
        random_orbit = harvest(model, feedback_seq, external_seqs)

        # Compare base and random orbits layer by layer
        for layer_name, base_states in base_orbit.items():
            for state_idx, (base_state, random_state) in enumerate(
                zip(base_states, random_orbit[layer_name])
            ):
                # Compute Euclidean distance over time
                norms_over_time = tf.norm(base_state - random_state, axis=-1)

                # Remove transient steps if specified
                if transient > 0:
                    norms_over_time = norms_over_time[:, transient:]

                # Δi = average distance over timesteps and batch
                delta = tf.reduce_mean(norms_over_time, axis=1)  # average per batch
                delta = tf.reduce_mean(delta)                    # average over batches

                # Accumulate for ESP index
                esp_indices[layer_name][state_idx] += tf.cast(delta, input_dtype)

                # Save history if requested
                if history:
                    esp_history[layer_name][state_idx] = (
                        esp_history[layer_name][state_idx]
                        .write(iter_idx, tf.cast(norms_over_time, input_dtype))
                    )
    print()  # newline after progress

    # --- Finalize indices by averaging over iterations ---
    for layer_name in esp_indices:
        esp_indices[layer_name] = [
            esp / iterations for esp in esp_indices[layer_name]
        ]

    # --- Restore original states ---
    set_reservoir_states(model, current_states)

    # --- Finalize history if requested ---
    if history:
        for layer_name in esp_history:
            esp_history[layer_name] = [
                tf.transpose(ta.stack(), perm=[0, 2, 1]) for ta in esp_history[layer_name]
            ]

    return (esp_indices, esp_history) if history else esp_indices

