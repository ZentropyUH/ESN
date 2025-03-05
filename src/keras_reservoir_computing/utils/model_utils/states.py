from typing import Tuple

import tensorflow as tf

import keras_reservoir_computing as krc
from keras_reservoir_computing.layers.reservoirs.base import BaseReservoir


def get_reservoir_states(model: tf.keras.Model) -> dict:
    """
    Get the states of the reservoirs in the model.

    Parameters
    ----------
    model : keras.Model
        The model containing the reservoirs.

    Returns
    -------
    dict
        A dictionary with keys as the names of the reservoir layers and values as the states of the reservoirs.
    """
    states = {}

    for layer in model.layers:
        if isinstance(layer, krc.layers.reservoirs.base.BaseReservoir):
            states[layer.name] = layer.get_states()

    return states


def set_reservoir_states(model: tf.keras.Model, states: dict) -> None:
    """
    Set the states of the reservoirs in the model.

    Parameters
    ----------
    model : keras.Model
        The model containing the reservoirs.
    states : dict
        A dictionary with keys as the names of the reservoir layers and values as the states of the reservoirs.

    Raises
    ------
    ValueError
        If the keys in the states dictionary do not match the names of the reservoir layers
    """

    # Check if the states are valid
    layer_names = [
        layer.name
        for layer in model.layers
        if isinstance(layer, krc.layers.reservoirs.base.BaseReservoir)
    ]

    if len(layer_names) == 0:
        return

    if set(states.keys()) != set(layer_names):
        raise ValueError(
            f"Invalid states. Expected keys in new states: {layer_names}. Got keys: {states.keys()}"
        )

    for layer in model.layers:
        if isinstance(layer, krc.layers.reservoirs.base.BaseReservoir):
            layer.set_states(states[layer.name])


def reset_reservoir_states(model: tf.keras.Model) -> None:
    """
    Reset the states of the reservoirs in the model.

    Parameters
    ----------
    model : keras.Model
        The model containing the reservoirs.
    """

    for layer in model.layers:
        if isinstance(layer, krc.layers.reservoirs.base.BaseReservoir):
            layer.reset_states()


def set_reservoir_random_states(model: tf.keras.Model, dist: str = "uniform") -> None:
    """
    Set the states of the reservoirs in the model to random values.

    Parameters
    ----------
    model : keras.Model
        The model containing the reservoirs.
    """

    for layer in model.layers:
        if isinstance(layer, krc.layers.reservoirs.base.BaseReservoir):
            layer.set_random_states(dist=dist)


@tf.function
def harvest(
    model: tf.keras.Model,
    feedback_seq: tf.Tensor,
    external_seqs: Tuple[tf.Tensor, ...] = (),
) -> dict:
    """
    Collect reservoir states while running the model with provided feedback sequence.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing BaseReservoir layers whose states we want to collect.
    feedback_seq : tf.Tensor
        The feedback sequence to use, shape [batch_size, timesteps, features].
    external_seqs : Tuple[tf.Tensor, ...]
        Tuple of external input sequences, each shaped [batch_size, timesteps, features].
        Should be in the same order as the model's input layers.

    Returns
    -------
    states_history : dict
        Dictionary of reservoir states for each BaseReservoir layer.

    Notes
    -----
    - The function uses a TensorFlow while loop to iterate over the feedback sequence.
    - The function records the states of BaseReservoir layers at each step of the feedback sequence
    and returns them in a dictionary.
    -
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


# def esp_index(
#     model: tf.keras.Model,
#     feedback_seq: tf.Tensor,
#     external_seqs: Tuple[tf.Tensor, ...] = (),
#     random_dist="uniform",
#     history: bool = False,
#     iterations: int = 10,
# ) -> dict:
#     """
#     Compute the Echo State Property (ESP) index for the reservoirs in the model.

#     Parameters
#     ----------
#     model : tf.keras.Model
#         The model containing BaseReservoir layers whose ESP index we want to compute.
#     feedback_seq : tf.Tensor
#         The feedback sequence to use, shape [batch_size, timesteps, features].
#     external_seqs : Tuple[tf.Tensor, ...]
#         Tuple of external input sequences, each shaped [batch_size, timesteps, features].
#         Should be in the same order as the model's input layers.
#     iterations : int
#         Number of iterations to use for computing the ESP index.

#     Returns
#     -------
#     esp_indices : dict
#         Dictionary of ESP indices for each BaseReservoir layer.

#     Notes
#     -----
#     - The function uses the harvest function to collect reservoir states while running the model
#     with the provided feedback sequence.
#     - The function computes the ESP index for each reservoir layer using the collected states.
#     """
#     # Save current states for restoring later
#     current_states = {
#         name: [tf.identity(state) for state in states]
#         for name, states in get_reservoir_states(model).items()
#     }

#     # Set zero states for the reservoirs to establish the base state orbit
#     reset_reservoir_states(model)

#     base_orbit = harvest(model, feedback_seq, external_seqs)

#     esp_indices = {key: [0.0 for state in states] for key, states in base_orbit.items()}

#     if history:
#         esp_history = {
#             key: [
#                 tf.TensorArray(
#                     dtype=tf.float32, size=int(base_orbit[key][i].shape[1])
#                 )  # Ensure integer size
#                 for i in range(len(states))
#             ]
#             for key, states in base_orbit.items()
#         }
#     for _ in range(iterations):

#         # Print progress
#         print(f"\rIteration {_ + 1}/{iterations}", end="", flush=True)

#         # Set random states for the reservoirs
#         set_reservoir_random_states(model, dist=random_dist)

#         # Collect states for the random state orbit
#         random_orbit = harvest(model, feedback_seq, external_seqs)

#         if history:
#             iter_history = {
#                 key: [
#                     tf.TensorArray(dtype=tf.float32, size=int(base_orbit[key][i].shape[1]))
#                     for i in range(len(states))
#                 ]
#                 for key, states in base_orbit.items()
#             }

#         for key in base_orbit.keys():
#             base_states = base_orbit[key]
#             random_states = random_orbit[key]

#             for i, (base_state, random_state) in enumerate(zip(base_states, random_states)):
#                 diff = base_state - random_state
#                 weights = tf.linspace(0.1, 1.0, num=tf.shape(base_state)[1])
#                 norms_over_time = tf.norm(diff, axis=-1)  # Norm at each time step

#                 if history:
#                     for t in range(tf.shape(norms_over_time)[1]):
#                         iter_history[key][i] = iter_history[key][i].write(t, norms_over_time[:, t])

#                 weighted_norm = tf.reduce_mean(weights * norms_over_time)
#                 esp_indices[key][i] += weighted_norm

#         if history:
#             # Convert TensorArrays to Tensors for this iteration
#             for key in iter_history.keys():
#                 for i in range(len(iter_history[key])):
#                     iter_history[key][i] = iter_history[key][i].stack()

#             # Initialize esp_history on the first iteration
#             if _ == 0:
#                 esp_history = {
#                     key: [tf.TensorArray(dtype=tf.float32, size=iterations) for _ in states]
#                     for key, states in base_orbit.items()
#                 }

#             # Store each iteration's history in esp_history
#             for key in esp_history.keys():
#                 for i in range(len(esp_history[key])):
#                     esp_history[key][i] = esp_history[key][i].write(_, iter_history[key][i])


#     # Average the ESP index over the iterations
#     for key in esp_indices.keys():
#         esp_indices[key] = [esp / iterations for esp in esp_indices[key]]

#     # Restore the original states
#     set_reservoir_states(model, current_states)

#     if history:
#         for key in esp_history.keys():
#             for i in range(len(esp_history[key])):
#                 esp_history[key][i] = esp_history[key][i].stack()  # Convert TensorArray to Tensor

#     return (esp_indices, esp_history) if history else esp_indices


def esp_index(
    model: tf.keras.Model,
    feedback_seq: tf.Tensor,
    external_seqs: Tuple[tf.Tensor, ...] = (),
    random_dist="uniform",
    history: bool = False,
    weighted: bool = False,
    iterations: int = 10,
) -> dict:
    """
    Compute the Echo State Property (ESP) index for the reservoirs in the model.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing BaseReservoir layers whose ESP index we want to compute.
    feedback_seq : tf.Tensor
        The feedback sequence to use, shape [batch_size, timesteps, features].
    external_seqs : Tuple[tf.Tensor, ...]
        Tuple of external input sequences, each shaped [batch_size, timesteps, features].
        Should be in the same order as the model's input layers.
    history : bool, optional
        Whether to return the full time evolution of norm differences.
    weighted : bool, optional
        Whether to use weighted norms for computing the ESP index.
    iterations : int
        Number of iterations to use for computing the ESP index.

    Returns
    -------
    esp_indices : dict
        Dictionary of ESP indices for each BaseReservoir layer.
    esp_history : dict, optional
        Dictionary of time evolution of norm differences (if history=True).
    """

    # Save current states for restoring later
    current_states = {
        name: [tf.identity(state) for state in states]
        for name, states in get_reservoir_states(model).items()
    }

    # Set zero states for the reservoirs to establish the base state orbit
    reset_reservoir_states(model)
    base_orbit = harvest(model, feedback_seq, external_seqs)

    # Initialize ESP index storage
    esp_indices = {key: [0.0 for _ in states] for key, states in base_orbit.items()}

    # Initialize history storage if required
    esp_history = (
        {
            key: [tf.TensorArray(dtype=tf.float32, size=iterations) for _ in states]
            for key, states in base_orbit.items()
        }
        if history
        else None
    )

    for iter_idx in range(iterations):
        print(f"\rIteration {iter_idx + 1}/{iterations}", end="", flush=True)

        set_reservoir_random_states(model, dist=random_dist)
        random_orbit = harvest(model, feedback_seq, external_seqs)

        for key, base_states in base_orbit.items():
            for i, (base_state, random_state) in enumerate(
                zip(base_states, random_orbit[key])
            ):
                diff = base_state - random_state
                norms_over_time = tf.norm(diff, axis=-1)

                if weighted:
                    weights = tf.linspace(
                        0.1, 1.0, num=tf.shape(base_state)[1]
                    )  # Increasing weights
                else:
                    weights = tf.ones_like(norms_over_time)

                weighted_norm = tf.reduce_mean(weights * norms_over_time)
                esp_indices[key][i] += weighted_norm

                if history:
                    esp_history[key][i] = esp_history[key][i].write(
                        iter_idx, norms_over_time
                    )

    # Average ESP indices over iterations
    for key in esp_indices:
        esp_indices[key] = [esp / iterations for esp in esp_indices[key]]

    set_reservoir_states(model, current_states)  # Restore original states

    if history:
        for key in esp_history:
            esp_history[key] = [
                tf.transpose(ta.stack(), perm=[0, 2, 1]) for ta in esp_history[key]
            ]  # Fix time axis

    return (esp_indices, esp_history) if history else esp_indices
