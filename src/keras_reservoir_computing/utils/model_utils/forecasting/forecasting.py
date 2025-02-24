import tensorflow as tf
from keras_reservoir_computing.layers.reservoirs.base import BaseReservoir
from typing import Tuple, Union, List


@tf.function()
def forecast(
    model: tf.keras.Model,
    initial_feedback: tf.Tensor,
    horizon: int = 1000,
    external_inputs: dict = {},
) -> Tuple[tf.Tensor, dict]:
    """
    Generate a forecast using a Keras model with feedback and external inputs.

    This function performs a multi-step forecast using a Keras model. The model
    is assumed to have multiple inputs, where the first input is the feedback
    input and the remaining inputs are external inputs. The model output at each
    step is fed back into the feedback input for the next step, enabling
    generative forecasting. The function also tracks and returns the internal states
    of any BaseReservoir layers within the model.

    Parameters
    ----------
    model : tf.keras.Model
        A Keras model whose first input is the feedback input and subsequent
        inputs are external inputs. It may contain BaseReservoir layers whose
        states will be tracked.
    external_inputs : dict
        A dictionary keyed by input_name (matching model input_names, skipping
        the feedback input), where each value is a tensor of shape
        [batch_size, horizon, input_dims_i].
    initial_feedback : tf.Tensor
        The initial feedback to start the forecast; shape should match what
        model.inputs[0] expects for a single step, e.g. [batch_size, 1, feedback_features].
    horizon : int
        Number of forecast steps to generate.

    Returns
    -------
    outputs : tf.Tensor
        A stacked time series of model outputs, typically of shape
        [batch_size, horizon, output_dim].
    states_history : dict
        A dictionary containing the history of reservoir states for each BaseReservoir
        layer in the model. The keys are the names of the BaseReservoir layers,
        and the values are tensors of shape [batch_size, horizon, state_dim].

    Notes
    -----
    - The model inputs are assumed to be in the following order: feedback input first,
      followed by external inputs.
    - The model inputs are starting from the timestep after the initial feedback.
    - Each external input is expected to have the shape [batch_size, horizon, input_dims].
    - The feedback input is updated at each step with the model's output from the previous step.
    - The function uses a TensorFlow while loop to iterate over the forecast horizon.
    - The function records the states of BaseReservoir layers at each step of the
      forecast and returns them in a dictionary.

    Examples
    --------
    >>> model = tf.keras.Model(inputs=[feedback_input, ext_input1, ext_input2], outputs=output)
    >>> external_inputs = {
    ...     'ext_input1': tf.random.normal([batch_size, horizon, input_dim1]),
    ...     'ext_input2': tf.random.normal([batch_size, horizon, input_dim2])
    ... }
    >>> initial_feedback = tf.random.normal([batch_size, 1, feedback_dim])
    >>> horizon = 10
    >>> outputs, states_history = forecast(model, external_inputs, initial_feedback, horizon)
    >>> print(outputs.shape)
    (batch_size, horizon, output_dim)
    >>> print(states_history['reservoir_layer_name'].shape)
    (batch_size, horizon, state_dim)
    """
    # The model inputs in order: feedback first, then exogenous inputs
    input_names = [
        _input.name for _input in model.inputs
    ]  # e.g. ["feedback_input", "ext_1", "ext_2", ...]
    if len(input_names) < 1:
        raise ValueError("Model must have at least one input (the feedback input).")


    batch_size = initial_feedback.shape[0]
    features = initial_feedback.shape[-1]

    horizon = tf.reduce_min(
        [horizon] + [tf.shape(ext)[1] for ext in external_inputs.values()]
    )

    # Skip the first one being the feedback input
    external_input_names = input_names[1:]  # The remainder are external inputs

    # Make sure user provided all external inputs required by the model
    for name in external_input_names:
        if name not in external_inputs:
            raise ValueError(
                f"Missing external input '{name}' in provided external inputs."
            )

    ext_tensors = [external_inputs[name] for name in external_input_names]
    ext_ta_list = [
        tf.TensorArray(dtype=ext.dtype, size=horizon, infer_shape=True).unstack(
            tf.transpose(ext, [1, 0, 2])
        )
        for ext in ext_tensors
    ]

    # Initialize the loop variables
    t0 = tf.constant(0, dtype=tf.int32)

    # We'll accumulate each output step in a TensorArray
    outputs_ta = tf.TensorArray(dtype=model.output.dtype, size=horizon)

    # Initialize states tracking
    states = {}
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            states[layer.name] = [
                tf.TensorArray(dtype=state.dtype, size=horizon, infer_shape=True)
                for state in layer.get_states()
            ]

    loop_vars = (t0, initial_feedback, outputs_ta, states)

    # Loop condition: t < horizon
    def cond(t, feedback, outputs_ta, states):
        return t < horizon

    # Loop body: slice each external input at time t, call model, update feedback
    def body(t, feedback, outputs_ta, states):
        # 1) Build a list of inputs for model call
        #    - feedback as the first
        #    - each external input sliced at time step t
        model_inputs = [feedback]

        model_inputs.extend(
            [tf.expand_dims(ext_ta.read(t), axis=1) for ext_ta in ext_ta_list]
        )

        # 2) Run a single forward pass
        out_t = model(model_inputs)  # shape [batch_size, 1, output_dim or ...]

        # 3) The output is the new feedback
        new_feedback = tf.reshape(out_t, [tf.shape(out_t)[0], 1, tf.shape(out_t)[-1]])

        # 4) Store the model output
        outputs_ta = outputs_ta.write(t, out_t)

        # 5) Store the reservoir states
        for layer in model.layers:
            if isinstance(layer, BaseReservoir):
                for i, (st_ta, st) in enumerate(
                    zip(states[layer.name], layer.get_states())
                ):
                    states[layer.name][i] = st_ta.write(t, st)

        # Print progress bar
        N = tf.maximum(horizon // 10, 1)  # Adjust frequency dynamically

        if tf.equal(t % N, 0):  # Only print every N steps
            progress = (t) * 100 // horizon
            progress_bar = tf.strings.reduce_join(tf.repeat("=", progress // 5))
            spaces_bar = tf.strings.reduce_join(tf.repeat(" ", 20 - (progress // 5)))

            progress_bar_full = tf.strings.join(["[", progress_bar, spaces_bar, "] "])
            tf.print(
                tf.strings.format(
                    "\rForecasting: {} / {} {}{}%",
                    (t + 1, horizon, progress_bar_full, progress),
                ),
                end="\r",
            )

        return t + 1, new_feedback, outputs_ta, states

    # If necessary, define shape invariants (especially for feedback)
    # Suppose feedback is shape: [batch_size, 1, ?]
    # We'll allow the last dimension to be unknown.
    shape_invariants = (
        t0.shape,  # Scalar, shape = ()
        tf.TensorShape([batch_size, 1, features]),  # feedback: [batch_size, 1, output_dim]
        tf.TensorShape(None),  # outputs_ta: TensorArray (dynamic shape)
        {  # states: dictionary of TensorArrays
            layer.name: [tf.TensorShape(None) for _ in layer.get_states()]
            for layer in model.layers
            if isinstance(layer, BaseReservoir)
        },
    )

    # Run tf.while_loop
    _, final_feedback, outputs_ta, states_history = tf.while_loop(
        cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants
    )

    # Stack time dimension: shape => [horizon, batch_size, 1, output_dim] typically
    outputs = outputs_ta.stack()
    # Possibly transpose to [batch_size, horizon, 1, output_dim]
    outputs = tf.transpose(outputs, [1, 0, 2, 3])  # or adapt if rank differs
    outputs = tf.squeeze(outputs, axis=2)  # Now (batch_size, horizon, features)

    # Stack reservoir states
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            for st_ta in states_history[layer.name]:
                states_history[layer.name] = st_ta.stack()
                states_history[layer.name] = tf.transpose(
                    states_history[layer.name], [1, 0, 2]
                )

    return outputs, states_history


def warmup_forecast(
    model: tf.keras.Model,
    warmup_data: Union[tf.Tensor, List[tf.Tensor]],
    forecast_data: Union[tf.Tensor, List[tf.Tensor]],
    horizon: int,
) -> tf.Tensor:
    """
    Runs a warmup phase using model.predict() on actual feedback, then switches
    to generative forecasting.

    Args:
        model : tf.keras.Model
            The trained model to use for prediction and forecasting.
        feedback_seq : tf.Tensor
            The full feedback sequence, shaped [batch, time, features].
        external_inputs : dict
            Dictionary of external inputs, each shaped [batch, time, features].
        warmup : int
            Number of initial time steps to run with true feedback.
        forecast_length : int
            How long to run generative forecasting if no external inputs exist.

    Returns:
        forecasted_output: tf.Tensor
            Forecasted sequence after the warmup phase.
    """

    input_names = [
        _input.name for _input in model.inputs
    ]  # e.g. ["feedback_input", "ext_1", "ext_2", ...]

    if not isinstance(forecast_data, list):
        batch_size = forecast_data.shape[0]
        initial_feedback = forecast_data[:, :1, :]
        external_inputs_post_warmup = {}
    else:
        batch_size = forecast_data[0].shape[0]
        initial_feedback = forecast_data[0][:, :1, :]
        external_inputs_post_warmup = {k: v for k, v in zip(input_names[1:], forecast_data[1:])}

    warmup_output = model.predict(warmup_data, batch_size=batch_size)

    # 6) Run forecast
    forecasted_output, states = forecast(
        model=model,
        external_inputs=external_inputs_post_warmup,
        initial_feedback=initial_feedback,
        horizon=horizon,
    )

    return forecasted_output, states


@tf.function
def harvest(
    model: tf.keras.Model, feedback_seq: tf.Tensor, external_seqs: dict = {}
) -> dict:
    """
    Collect reservoir states while running the model with provided feedback sequence.

    Parameters
    ----------
    model : tf.keras.Model
        The model containing BaseReservoir layers whose states we want to collect.
    feedback_seq : tf.Tensor
        The feedback sequence to use, shape [batch_size, timesteps, features].
    external_seqs : dict
        Dictionary of external input sequences, each shaped [batch_size, timesteps, features].

    Returns
    -------
    states_history : dict
        Dictionary of reservoir states for each BaseReservoir layer.
    """
    input_names = [_input.name for _input in model.inputs]
    if len(input_names) < 1:
        raise ValueError("Model must have at least one input (the feedback input).")

    # should be the minimum of the feedback sequence length and the input sequences length
    horizon = tf.reduce_min(
        [tf.shape(feedback_seq)[1]] + [tf.shape(v)[1] for v in external_seqs.values()]
    )

    # First one being the feedback input
    external_input_names = input_names[1:]

    for name in external_input_names:
        if name not in external_seqs:
            raise ValueError(
                f"Missing external input '{name}' in provided external inputs."
            )

    # Slices optimization
    feedback_ta = tf.TensorArray(dtype=feedback_seq.dtype, size=horizon)
    feedback_ta = feedback_ta.unstack(
        tf.transpose(feedback_seq, [1, 0, 2])
    )  # Shape [timesteps, batch, features]

    # Optimize external inputs slicing
    ext_tensors = [external_seqs[name] for name in external_input_names]
    ext_ta_list = [
        tf.TensorArray(dtype=ext.dtype, size=horizon, infer_shape=True).unstack(
            tf.transpose(ext, [1, 0, 2])
        )
        for ext in ext_tensors
    ]

    # Initialize the loop variables
    t0 = tf.constant(0, dtype=tf.int32)

    # Initialize states tracking
    states = {}
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            states[layer.name] = [
                tf.TensorArray(dtype=state.dtype, size=horizon, infer_shape=True)
                for state in layer.get_states()
            ]

    loop_vars = (t0, states)

    # Loop condition: t < horizon
    def cond(t, states):
        return t < horizon

    def body(t, states):
        # Build inputs list with actual feedback and external inputs
        model_inputs = [
            tf.expand_dims(feedback_ta.read(t), axis=1)
        ]  # Now (batch, 1, features)

        model_inputs.extend(
            [tf.expand_dims(ext_ta.read(t), axis=1) for ext_ta in ext_ta_list]
        )

        # Run forward pass
        _ = model(model_inputs)

        # Store reservoir states
        for layer in model.layers:
            if isinstance(layer, BaseReservoir):
                for i, (st_ta, st) in enumerate(
                    zip(states[layer.name], layer.get_states())
                ):
                    states[layer.name][i] = st_ta.write(t, st)

        # Print progress bar
        N = tf.maximum(horizon // 10, 1)  # Adjust frequency dynamically

        if tf.equal(t % N, 0):  # Only print every N steps
            progress = (t) * 100 // horizon
            progress_bar = tf.strings.reduce_join(tf.repeat("=", progress // 5))
            spaces_bar = tf.strings.reduce_join(tf.repeat(" ", 20 - (progress // 5)))

            progress_bar_full = tf.strings.join(["[", progress_bar, spaces_bar, "] "])
            tf.print(
                tf.strings.format(
                    "\rForecasting: {} / {} {}{}%",
                    (t + 1, horizon, progress_bar_full, progress),
                ),
                end="\r",
            )

        return t + 1, states

    shape_invariants = (
        t0.shape,
        {
            layer.name: [tf.TensorShape(None) for _ in layer.get_states()]
            for layer in model.layers
            if isinstance(layer, BaseReservoir)
        },
    )

    # Run the loop
    _, states_history = tf.while_loop(
        cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants
    )

    # Stack states
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            for i, st_ta in enumerate(states_history[layer.name]):
                states_history[layer.name][i] = tf.transpose(st_ta.stack(), [1, 0, 2])

    return states_history
