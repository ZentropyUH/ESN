"""Forecasting utilities for reservoir computing models.

This module provides functions for generating forecasts with trained reservoir models,
including multi-step auto-regressive forecasting and warmup-based forecasting.
"""

from typing import Dict, List, Tuple, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.base import BaseReservoir


@tf.function
def forecast(
    model: tf.keras.Model,
    initial_feedback: tf.Tensor,
    horizon: int = 1000,
    external_inputs: Tuple[tf.Tensor, ...] = (),
) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
    """
    Generate a forecast using a trained model with auto-regressive feedback.

    This function performs a multi-step forecast using a Keras model. The model's output
    at each step is fed back as input for the next step, enabling auto-regressive forecasting.
    The function also optionally tracks and returns the internal states of reservoir layers.

    Parameters
    ----------
    model : tf.keras.Model
        A Keras model whose first input is the feedback input and subsequent
        inputs are optional external inputs.
    initial_feedback : tf.Tensor
        The initial feedback to start the forecast; shape should be
        [batch_size, 1, feedback_features].
    horizon : int, optional
        Number of time steps to forecast. Default is 1000.
    external_inputs : Tuple[tf.Tensor, ...], optional
        External inputs for each time step, each shaped [batch_size, horizon, features].
        Must match the number of external inputs expected by the model.

    Returns
    -------
    outputs : tf.Tensor
        Forecasted time series with shape [batch_size, horizon, output_features].
    states_history : Dict[str, List[tf.Tensor]]
        Dictionary of reservoir states history. Keys are reservoir layer names,
        values are lists of tensors with shape [batch_size, horizon, state_dim].

    Raises
    ------
    ValueError
        If the model has no inputs or the number of external inputs doesn't match
        what the model expects.
    """
    # Validate inputs
    input_names = [_input.name for _input in model.inputs]
    if len(input_names) < 1:
        raise ValueError("Model must have at least one input (the feedback input).")

    # Extract shapes for validation and initialization
    batch_size = initial_feedback.shape[0]
    features = initial_feedback.shape[-1]

    # Limit horizon to the available external data length if provided
    if external_inputs:
        min_ext_length = tf.reduce_min([ext.shape[1] for ext in external_inputs])
        horizon = tf.minimum(horizon, min_ext_length)

    # Validate that we have the right number of external inputs
    external_input_count = len(input_names) - 1
    if len(external_inputs) != external_input_count:
        raise ValueError(
            f"Expected {external_input_count} external inputs, but got {len(external_inputs)}."
        )

    # Initialize loop variables
    t0 = tf.constant(0, dtype=tf.int32)
    outputs_ta = tf.TensorArray(
        dtype=model.output.dtype, size=horizon, element_shape=[batch_size, features]
    )
    states: Dict[str, List[tf.TensorArray]] = {}

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
        # Prepare model inputs based on whether we have external inputs
        if not external_inputs:
            model_inputs = feedback
        else:
            model_inputs = [feedback] + [
                tf.expand_dims(ext[:, t, :], axis=1) for ext in external_inputs
            ]

        # Run the model for this timestep
        out_t = model(model_inputs)

        # Prepare feedback for next step (reshape ensures correct dimensions)
        new_feedback = tf.reshape(out_t, [batch_size, 1, features])

        # Store the output
        outputs_ta = outputs_ta.write(t, tf.squeeze(out_t, axis=1))

        # Track reservoir states
        for layer in model.layers:
            if isinstance(layer, BaseReservoir):
                for i, (st_ta, st) in enumerate(
                    zip(states[layer.name], layer.get_states())
                ):
                    states[layer.name][i] = st_ta.write(t, st)

        # Report progress at sensible intervals
        progress_interval = tf.maximum(horizon // 10, 100)
        if tf.equal(t % progress_interval, 0) or tf.equal(t, horizon - 1):
            tf.print(
                "\rForecasting step:",
                t,
                "of",
                horizon,
                "[",
                (t + 1) * 100 // horizon,
                "%]",
                end="\r",
            )

        return t + 1, new_feedback, outputs_ta, states

    # Define shape invariants for the tf.while_loop to help TensorFlow optimize
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

    # Run the forecast loop - use parallel_iterations=1 for sequential execution
    # which is necessary for correct state tracking in recurrent models
    _, _, outputs_ta, states_history = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars,
        shape_invariants=shape_invariants,
        parallel_iterations=1,
    )

    # Extract and reshape outputs to [batch, time, features]
    outputs = outputs_ta.stack()
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Process reservoir states if tracked
    for layer in model.layers:
        if isinstance(layer, BaseReservoir):
            layer_states = []
            for st_ta in states_history[layer.name]:
                state = st_ta.stack()
                state = tf.transpose(state, [1, 0, 2])  # [batch, time, state_dim]
                layer_states.append(state)
            # Store all states for this layer
            states_history[layer.name] = layer_states

    # Print newline to clear the progress indicator
    tf.print()

    return outputs, states_history


def warmup_forecast(
    model: tf.keras.Model,
    warmup_data: Union[tf.Tensor, List[tf.Tensor]],
    forecast_data: Union[tf.Tensor, List[tf.Tensor]],
    horizon: int,
) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
    """
    Run a warmup phase on actual data before auto-regressive forecasting.

    This function first initializes reservoir states by running the model on actual data
    (warmup phase), then switches to auto-regressive forecasting. This approach typically
    produces better forecasts by ensuring the model starts from meaningful states.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model to use for prediction and forecasting.
    warmup_data : Union[tf.Tensor, List[tf.Tensor]]
        Data to use for initializing the model's internal states. If a list,
        the first element should be the primary input and the rest are external inputs.
    forecast_data : Union[tf.Tensor, List[tf.Tensor]]
        Initial data for the forecasting phase. If a list, the first element should
        be the primary input and the rest are external inputs for the forecast period.
    horizon : int
        Number of forecast steps to generate.
    batch_size : Optional[int], optional
        Batch size to use for the warmup phase. If None, automatically determined
        from input data. Default is None.

    Returns
    -------
    forecasted_output : tf.Tensor
        Forecasted sequence with shape [batch_size, horizon, features].
    states : Dict[str, List[tf.Tensor]]
        Dictionary of reservoir states.

    Raises
    ------
    ValueError
        If the number of inputs provided doesn't match what the model expects.
    """
    input_names = [_input.name for _input in model.inputs]
    input_count = len(input_names)

    # Validate inputs
    if isinstance(warmup_data, list) and len(warmup_data) != input_count:
        raise ValueError(
            f"Expected {input_count} inputs for warmup_data, but got {len(warmup_data)}."
        )

    if isinstance(forecast_data, list) and len(forecast_data) != input_count:
        raise ValueError(
            f"Expected {input_count} inputs for forecast_data, but got {len(forecast_data)}."
        )

    # Process inputs based on whether we have a single tensor or list of tensors
    if isinstance(forecast_data, list):
        # Multiple tensors: primary input + external inputs
        auto_batch_size = tf.shape(forecast_data[0])[0]
        initial_feedback = forecast_data[0][:, :1, :]
        external_inputs = tuple(forecast_data[1:])
    else:
        # Single tensor: only primary input
        auto_batch_size = tf.shape(forecast_data)[0]
        initial_feedback = forecast_data[:, :1, :]
        external_inputs = ()

    # Perform warmup to initialize reservoir states
    print("Warming up model with data...")
    _ = model.predict(warmup_data, batch_size=auto_batch_size, verbose=0)

    # Run forecast with initialized states
    print(f"Running forecast for {horizon} steps...")
    forecasted_output, states = forecast(
        model=model,
        initial_feedback=initial_feedback,
        external_inputs=external_inputs,
        horizon=horizon,
    )

    print(f"Forecast completed: output shape {forecasted_output.shape}")
    return forecasted_output, states
