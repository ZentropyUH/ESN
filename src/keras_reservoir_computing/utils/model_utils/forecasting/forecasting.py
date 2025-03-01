import tensorflow as tf
from keras_reservoir_computing.layers.reservoirs.base import BaseReservoir
from typing import Tuple, Union, List


@tf.function()
def forecast(
    model: tf.keras.Model,
    initial_feedback: tf.Tensor,
    horizon: int = 1000,
    external_inputs: Tuple[tf.Tensor, ...] = (),
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
    external_inputs : Tuple[tf.Tensor, ...]
        A tuple of tensors, each shaped [batch_size, horizon, input_dims_i], in
        the same order as the model's external inputs.
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
    """
    input_names = [_input.name for _input in model.inputs]
    if len(input_names) < 1:
        raise ValueError("Model must have at least one input (the feedback input).")

    batch_size = initial_feedback.shape[0]
    features = initial_feedback.shape[-1]

    horizon = tf.reduce_min([horizon] + [tf.shape(ext)[1] for ext in external_inputs])

    external_input_count = len(input_names) - 1
    if len(external_inputs) != external_input_count:
        raise ValueError(
            f"Expected {external_input_count} external inputs, but got {len(external_inputs)}."
        )

    t0 = tf.constant(0, dtype=tf.int32)
    outputs_ta = tf.TensorArray(dtype=model.output.dtype, size=horizon)
    states = {}
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            states[layer.name] = [
                tf.TensorArray(dtype=state.dtype, size=horizon, infer_shape=True)
                for state in layer.get_states()
            ]

    loop_vars = (t0, initial_feedback, outputs_ta, states)

    def cond(t, feedback, outputs_ta, states):
        return t < horizon

    def body(t, feedback, outputs_ta, states):
        model_inputs = [feedback]
        if external_inputs:
            model_inputs.extend(
                [tf.expand_dims(ext[:, t, :], axis=1) for ext in external_inputs]
            )

        model_inputs = model_inputs[0] if len(model_inputs) == 1 else model_inputs

        out_t = model(model_inputs)
        new_feedback = tf.reshape(out_t, [tf.shape(out_t)[0], 1, tf.shape(out_t)[-1]])
        outputs_ta = outputs_ta.write(t, out_t)

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
                    (t, horizon, progress_bar_full, progress),
                ),
                end="\r",
            )

        return t + 1, new_feedback, outputs_ta, states

    shape_invariants = (
        t0.shape,
        tf.TensorShape([batch_size, 1, features]),
        tf.TensorShape(None),
        {
            layer.name: [tf.TensorShape(None) for _ in layer.get_states()]
            for layer in model.layers
            if isinstance(layer, BaseReservoir)
        },
    )

    _, final_feedback, outputs_ta, states_history = tf.while_loop(
        cond=cond, body=body, loop_vars=loop_vars, shape_invariants=shape_invariants
    )

    outputs = outputs_ta.stack()
    outputs = tf.transpose(outputs, [1, 0, 2, 3])
    outputs = tf.squeeze(outputs, axis=2)

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
        warmup_data : Union[tf.Tensor, List[tf.Tensor]]
            Warmup data to use for the initial phase of the forecast. If a list,
            the first element should be the feedback data and the rest should be
            external inputs.
        forecast_data : Union[tf.Tensor, List[tf.Tensor]]
            Forecast data to use for the generative phase of the forecast. If a list,
            the first element should be the feedback data and the rest should be
            external inputs.
        horizon : int
            Number of forecast steps to generate.
    Returns:
        forecasted_output: tf.Tensor
            Forecasted sequence after the warmup phase.
    """

    input_names = [
        _input.name for _input in model.inputs
    ]  # e.g. ["feedback_input", "ext_1", "ext_2", ...]
    
    if isinstance(forecast_data, list) and len(forecast_data) != len(input_names):
        raise ValueError(
            f"Expected {len(input_names)} total inputs, but got {len(forecast_data)}."
        )

    if not isinstance(forecast_data, list): # Single tensor: feedback only
        batch_size = forecast_data.shape[0]
        initial_feedback = forecast_data[:, :1, :]
        external_inputs_post_warmup = ()
    else: # Multiple tensors: feedback + external inputs
        batch_size = forecast_data[0].shape[0]
        initial_feedback = forecast_data[0][:, :1, :]
        external_inputs_post_warmup = tuple(forecast_data[1:])

    _ = model.predict(warmup_data, batch_size=batch_size, verbose=0)

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
    model: tf.keras.Model, feedback_seq: tf.Tensor, external_seqs: Tuple[tf.Tensor, ...] = ()
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
