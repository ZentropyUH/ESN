"""Forecasting utilities for reservoir-computing models.

This module provides high-level helpers that turn an already-trained
reservoir-computing network into a *multi-step* forecaster.

Two complementary workflows are exposed:

* :func:`forecast` - **pure auto-regressive** forecasting that recursively
  feeds back the model's own output.
* :func:`warmup_forecast` - **warm-start + auto-regressive** forecasting that
  first *warms up* the model on ground-truth data to initialise meaningful
  reservoir states and then switches to :func:`forecast`.

Both routines are fully graph-compatible (decorated with
:pyfunc:`tf.function`) and can optionally record the hidden states emitted by
any :class:`keras_reservoir_computing.layers.reservoirs.layers.base.BaseReservoir`
layer contained in the network.

Example
-------
>>> outputs, states = forecast(model, init_fb, horizon=500)
>>> outputs.shape
TensorShape([batch, 500, output_features])

>>> outputs, states = warmup_forecast(model, warmup_data, forecast_data, horizon=500, show_progress=True)
>>> outputs.shape
TensorShape([batch, 500, output_features])
"""

import weakref
from typing import Callable, Dict, List, Tuple, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.layers.base import BaseReservoir
from keras_reservoir_computing.utils.tensorflow import (
    suppress_retracing_during_call,
    tf_function,
)

__all__ = [
    "warmup_forecast",
]

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
        predict_factory._cache = weakref.WeakKeyDictionary()

    if model in predict_factory._cache:
        return predict_factory._cache[model]

    @tf_function(reduce_retracing=True, jit_compile=True)
    def predict(data: tf.Tensor) -> tf.Tensor:
        """Predict the output of the model.

        Args:
            data: The input data to the model.

        Returns:
            The output of the model.
        """
        return model(data, training=False)

    predict_factory._cache[model] = predict
    return predict

def forecast_states_factory(model: tf.keras.Model) -> Callable:

    if not hasattr(forecast_states_factory, "_cache"):
        forecast_states_factory._cache = weakref.WeakKeyDictionary()

    if model in forecast_states_factory._cache:
        return forecast_states_factory._cache[model]

    @tf_function(reduce_retracing=True, jit_compile=True)
    def forecast(
        initial_feedback: tf.Tensor,
        horizon: int = 1000,
        external_inputs: Tuple[tf.Tensor, ...] = (),
    ) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
        """Auto-regressive multi-step forecast.

        Starting from an *initial* feedback vector, the function repeatedly calls
        ``model`` for ``horizon`` steps.  At every step the freshly predicted
        output is **fed back** as first input for the next call - standard
        auto-regressive inference.  Optional *exogenous* inputs can be supplied
        in lock-step with the forecast horizon.

        The routine is decorated with :pyfunc:`tf.function`, hence it runs as a
        compiled TensorFlow graph while still returning ordinary :class:`tf.Tensor`
        objects.

        Parameters
        ----------
        initial_feedback
            Tensor of shape ``[batch, 1, feedback_features]`` that seeds the first
            forecast step.
        horizon
            Number of forecast steps to generate (default: ``1000``).
        external_inputs
            Tuple containing one tensor *per external input* expected by
            ``model``.  Each tensor must have shape
            ``[batch, horizon, ext_features]``.  Pass an empty tuple (default) if
            no external inputs are required.

        Returns
        -------
        outputs
            Tensor with shape ``[batch, horizon, output_features]`` holding the
            forecast.
        states_history
            Mapping ``layer_name -> list[state_tensor]`` where each
            ``state_tensor`` has shape ``[batch, horizon, state_dim]``.  Only
            layers that subclass :class:`BaseReservoir` are tracked.

        Raises
        ------
        ValueError
            If input shapes are inconsistent or ``external_inputs`` does not match
            the model signature.

        Notes
        -----
        *Parallelism* - ``parallel_iterations`` is set to ``1`` inside the
        ``tf.while_loop`` to preserve sequential semantics and consistent hidden
        state tracking.  This sacrifices some graph-level parallelism in exchange
        for correctness.

        Examples
        --------
        >>> outputs, states = forecast(model, init_fb, horizon=200)
        >>> outputs.shape  # doctest: +ELLIPSIS
        TensorShape([..., 200, output_features])
        """
        # Validate inputs
        input_names = [_input.name for _input in model.inputs]
        if len(input_names) < 1:
            raise ValueError("Model must have at least one input (the feedback input).")

        if len(initial_feedback.shape) != 3 or initial_feedback.shape[1] != 1:
            raise ValueError(
                "Expected initial_feedback shape [batch_size, 1, features], but got "
                f"{initial_feedback.shape}"
            )

        batch_size = initial_feedback.shape[0]
        features = initial_feedback.shape[-1]

        # Convert horizon to a tensor to avoid retracing
        horizon_const = tf.convert_to_tensor(horizon, dtype=tf.int32)

        # If external inputs are present, constrain the horizon to their length
        if external_inputs:
            ext_lengths   = [tf.shape(ext)[1] for ext in external_inputs]
            min_ext_len   = tf.reduce_min(ext_lengths)
            horizon_t     = tf.minimum(horizon_const, min_ext_len)
        else:
            horizon_t     = horizon_const

        external_input_count = len(input_names) - 1
        if len(external_inputs) != external_input_count:
            raise ValueError(
                f"Expected {external_input_count} external inputs, but got {len(external_inputs)}."
            )

        # Shape consistency for external inputs
        for i, ext_input in enumerate(external_inputs):
            if len(ext_input.shape) != 3:
                raise ValueError(
                    f"Expected external input {i} to have shape [batch_size, horizon, features], "
                    f"but got {ext_input.shape}"
                )
            if ext_input.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: initial_feedback has batch size {batch_size}, "
                    f"but external input {i} has batch size {ext_input.shape[0]}"
                )

        # TensorArrays for outputs and (optionally) states
        t0 = tf.constant(0, dtype=tf.int32)
        outputs_ta = tf.TensorArray(dtype=model.output.dtype,
                                    size=horizon_t,
                                    element_shape=[batch_size, features])
        states: Dict[str, List[tf.TensorArray]] = {}

        for layer in model.layers:
            if isinstance(layer, BaseReservoir):
                states[layer.name] = [
                    tf.TensorArray(dtype=state.dtype, size=horizon_t, infer_shape=True)
                    for state in layer.get_states()
                ]

        loop_vars = (t0, initial_feedback, outputs_ta, states)

        def cond(t, *_):  # noqa: D401 - short lambda-style condition OK
            return t < horizon_t

        def body(t, feedback, outputs_ta, states):  # noqa: D401 - internal helper
            # Assemble current inputs
            model_inputs = feedback if not external_inputs else [feedback] + [
                tf.expand_dims(ext[:, t, :], axis=1) for ext in external_inputs
            ]

            out_t = model(model_inputs)
            new_feedback = tf.reshape(out_t, [batch_size, 1, features])
            outputs_ta = outputs_ta.write(t, tf.squeeze(out_t, axis=1))

            # Track reservoir states
            for layer in model.layers:
                if isinstance(layer, BaseReservoir):
                    for i, (st_ta, st) in enumerate(zip(states[layer.name], layer.get_states())):
                        states[layer.name][i] = st_ta.write(t, st)

            return t + 1, new_feedback, outputs_ta, states

        shape_invariants = (
            t0.shape,
            tf.TensorShape([batch_size, 1, features]),
            tf.TensorShape(None),
            {ln: [tf.TensorShape(None) for _ in lst] for ln, lst in states.items()},
        )

        _, _, outputs_ta, states_history = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            shape_invariants=shape_invariants,
            parallel_iterations=1,
        )

        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])  # [batch, time, features]

        for layer in model.layers:
            if isinstance(layer, BaseReservoir):
                layer_states = []
                for st_ta in states_history[layer.name]:
                    state = tf.transpose(st_ta.stack(), [1, 0, 2])
                    layer_states.append(state)
                states_history[layer.name] = layer_states

        return outputs, states_history

    forecast_states_factory._cache[model] = forecast
    return forecast

def forecast_no_states_factory(model: tf.keras.Model) -> Callable:
    if not hasattr(forecast_no_states_factory, "_cache"):
        forecast_no_states_factory._cache = weakref.WeakKeyDictionary()

    if model in forecast_no_states_factory._cache:
        return forecast_no_states_factory._cache[model]

    @tf_function(reduce_retracing=True, jit_compile=True)
    def forecast_no_states(
        initial_feedback: tf.Tensor,
        horizon: int = 1000,
        external_inputs: Tuple[tf.Tensor, ...] = (),
    ) -> tf.Tensor:
        """Auto-regressive forecast without state tracking."""

        input_names = [_input.name for _input in model.inputs]
        if len(input_names) < 1:
            raise ValueError("Model must have at least one input (the feedback input).")

        if len(initial_feedback.shape) != 3 or initial_feedback.shape[1] != 1:
            raise ValueError(
                "Expected initial_feedback shape [batch, 1, features], but got "
                f"{initial_feedback.shape}")

        batch_size = initial_feedback.shape[0]
        features = initial_feedback.shape[-1]

        # Convert horizon to a tensor to avoid retracing
        horizon_const = tf.convert_to_tensor(horizon, dtype=tf.int32)

        # If external inputs are present, constrain the horizon to their length
        if external_inputs:
            ext_lengths = [tf.shape(ext)[1] for ext in external_inputs]
            min_ext_len   = tf.reduce_min(ext_lengths)
            horizon_t     = tf.minimum(horizon_const, min_ext_len)
        else:
            horizon_t = horizon_const


        external_input_count = len(input_names) - 1
        if len(external_inputs) != external_input_count:
            raise ValueError(
                f"Expected {external_input_count} external inputs, but got {len(external_inputs)}."
            )

        # Shape consistency for external inputs
        for i, ext_input in enumerate(external_inputs):
            if len(ext_input.shape) != 3:
                raise ValueError(
                    f"Expected external input {i} to have shape [batch_size, horizon, features], "
                    f"but got {ext_input.shape}"
                )
            if ext_input.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: initial_feedback has batch size {batch_size}, "
                    f"but external input {i} has batch size {ext_input.shape[0]}"
                )

        # TensorArrays for outputs and (optionally) states
        t0 = tf.constant(0, dtype=tf.int32)
        outputs_ta = tf.TensorArray(dtype=model.output.dtype,
                                    size=horizon_t,
                                    element_shape=[batch_size, features])

        loop_vars = (t0, initial_feedback, outputs_ta)

        def cond(t, *_):
            return t < horizon_t

        def body(t, feedback, outputs_ta):
            model_inputs = feedback if not external_inputs else [feedback] + [
                tf.expand_dims(ext[:, t, :], axis=1) for ext in external_inputs
            ]

            out_t = model(model_inputs)
            new_feedback = tf.reshape(out_t, [batch_size, 1, features])
            outputs_ta = outputs_ta.write(t, tf.squeeze(out_t, axis=1))

            return t + 1, new_feedback, outputs_ta

        shape_invariants = (
            t0.shape,
            tf.TensorShape([batch_size, 1, features]),
            tf.TensorShape(None),
        )

        _, _, outputs_ta = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            shape_invariants=shape_invariants,
            parallel_iterations=1,
        )

        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])
        return outputs

    forecast_no_states_factory._cache[model] = forecast_no_states
    return forecast_no_states


@suppress_retracing_during_call
def warmup_forecast(
    model: tf.keras.Model,
    warmup_data: Union[tf.Tensor, List[tf.Tensor]],
    forecast_data: Union[tf.Tensor, List[tf.Tensor]],
    horizon: int,
    show_progress: bool = True,
    states: bool = False,
) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
    """Warm-start the model then forecast auto-regressively.

    The function executes in two clearly separated phases:

    1. **Warm-up** - The model is driven by *ground-truth* ``warmup_data``
       (teacher forcing) to push internal reservoir states onto a realistic
       trajectory.
    2. **Forecast** - The routine switches to :func:`forecast` using
       ``forecast_data`` as initial feedback (and optional exogenous inputs)
       to produce ``horizon`` steps into the future.

    Parameters
    ----------
    model
        The trained Keras model.
    warmup_data
        Tensor *or* list of tensors fed to ``model`` during the warm-up pass.
        The cardinality must equal ``len(model.inputs)``.
    forecast_data
        Data that seeds the auto-regressive phase.  Follows the same structure
        as ``warmup_data``.  The first element/tensor must contain at least one
        time step because ``forecast`` will extract
        ``forecast_data[0][:, :1, :]`` as *initial feedback*.
    horizon
        Number of auto-regressive steps to generate **after** warm-up.
    show_progress
        If *True* prints progress indicators for both phases (default).
    states
        If *True* (default: *False*) the hidden-state history is returned.

    Returns
    -------
    forecasted_output
        Tensor with shape ``[batch, horizon, output_features]`` containing the
        prediction.
    states
        Hidden-state history exactly as returned by :func:`forecast` if ``states`` is *True*.

    Raises
    ------
    ValueError
        If ``warmup_data`` or ``forecast_data`` are incompatible with the model
        signature.

    Examples
    --------
    >>> _ = model.fit(x_train, y_train)  # model already trained
    >>> y_pred, _ = warmup_forecast(model, x_val, x_seed, horizon=300)
    """
    input_count = len(model.inputs)

    # Basic signature checks -------------------------------------------------
    if isinstance(warmup_data, list) and len(warmup_data) != input_count:
        raise ValueError(
            f"Expected {input_count} inputs for warmup_data, but got {len(warmup_data)}."
        )
    if isinstance(forecast_data, list) and len(forecast_data) != input_count:
        raise ValueError(
            f"Expected {input_count} inputs for forecast_data, but got {len(forecast_data)}."
        )

    # ----------------------------------------------------------------------
    # Split forecast_data into feedback + optional exogenous inputs
    # ----------------------------------------------------------------------
    if isinstance(forecast_data, list):
        initial_feedback = forecast_data[0][:, :1, :]
        external_inputs: Tuple[tf.Tensor, ...] = tuple(forecast_data[1:])
        batch_size = tf.shape(forecast_data[0])[0]

        # Ensure batch size consistency across exogenous inputs
        for i, ext in enumerate(external_inputs):
            if tf.shape(ext)[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: main input has batch size {batch_size}, "
                    f"but external input {i} has batch size {tf.shape(ext)[0]}"
                )
    else:
        initial_feedback = forecast_data[:, :1, :]
        external_inputs = ()
        batch_size = tf.shape(forecast_data)[0]

    # ----------------------------------------------------------------------
    # Phase 1 - warm-up
    # ----------------------------------------------------------------------
    if show_progress:
        print("Warming up model with teacher-forced data…")
    # _ = model.predict(warmup_data, batch_size=batch_size, verbose=1 if show_progress else 0)
    predict_fn = predict_factory(model)
    _ = predict_fn(warmup_data)

    # ----------------------------------------------------------------------
    # Phase 2 - auto-regressive forecast
    # ----------------------------------------------------------------------
    if show_progress:
        print(f"Running auto-regressive forecast for {horizon} steps…")

    if states:
        forecast_fn = forecast_states_factory(model)
        forecasted_output, states_history = forecast_fn(
            initial_feedback=initial_feedback,
            external_inputs=external_inputs,
            horizon=horizon,
        )
    else:
        forecast_fn = forecast_no_states_factory(model)
        forecasted_output = forecast_fn(
            initial_feedback=initial_feedback,
            external_inputs=external_inputs,
            horizon=horizon,
        )
        states_history = {}

    # Delete reference to potentially large objects
    # del forecast_fn
    # Clear TensorFlow's memory
    # tf.keras.backend.clear_session()

    if show_progress:
        print(f"Forecast completed - output shape {forecasted_output.shape}")

    return forecasted_output, states_history
