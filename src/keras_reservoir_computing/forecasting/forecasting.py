import sys
import weakref
from typing import Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.layers.base import BaseReservoir
from keras_reservoir_computing.utils.tensorflow import tf_function

# -----------------------------------------------------------------------------
# Module-level caches
# -----------------------------------------------------------------------------
_FORECAST_CACHE: "weakref.WeakKeyDictionary[tf.keras.Model, Dict[Tuple[bool, bool], Callable]]" = weakref.WeakKeyDictionary()
_PREDICT_CACHE: "weakref.WeakKeyDictionary[tf.keras.Model, Callable]" = weakref.WeakKeyDictionary()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _validate_inputs(
    initial_feedback: tf.Tensor,
    external_inputs: Tuple[tf.Tensor, ...],
    external_input_count: int,
) -> Tuple[int, int, tf.Tensor]:
    """Validate input shapes and return (batch_size, features, horizon_t)."""
    if len(initial_feedback.shape) != 3 or initial_feedback.shape[1] != 1:
        raise ValueError(
            f"Expected initial_feedback [batch, 1, features], got {initial_feedback.shape}"
        )

    if len(external_inputs) != external_input_count:
        raise ValueError(
            f"Expected {external_input_count} external inputs, got {len(external_inputs)}"
        )

    batch_size = initial_feedback.shape[0]
    features = initial_feedback.shape[-1]

    for i, ext in enumerate(external_inputs):
        if len(ext.shape) != 3:
            raise ValueError(
                f"External input {i} must have rank-3 [batch, time, features], got {ext.shape}"
            )
        if ext.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: feedback batch {batch_size}, ext[{i}] batch {ext.shape[0]}"
            )

    return batch_size, features


def _setup_state_arrays(
    reservoir_layers: List[BaseReservoir],
    horizon_t: tf.Tensor,
) -> Tuple[Dict[str, List[tf.TensorArray]], Dict[str, List[tf.TensorShape]]]:
    """Create TensorArrays for each reservoir state."""
    if not reservoir_layers:
        return {}, {}

    states_init = {}
    shape_invariants = {}

    for layer in reservoir_layers:
        init_list, shape_list = [], []
        for st in layer.get_states():
            ta = tf.TensorArray(
                dtype=st.dtype,
                size=horizon_t,
                element_shape=st.shape,  # fixed element_shape for perf
            )
            init_list.append(ta)
            shape_list.append(tf.TensorShape(None))
        states_init[layer.name] = init_list
        shape_invariants[layer.name] = shape_list

    return states_init, shape_invariants


# -----------------------------------------------------------------------------
# Core forecast factory
# -----------------------------------------------------------------------------
def forecast_factory(
    model: tf.keras.Model,
    *,
    track_states: bool = False,
    jit_compile: bool = True,
    show_progress: bool = True,
) -> Callable:
    """
    Return a compiled auto-regressive forecast function for `model`.

    The returned function has signature:
        forecast(initial_feedback, horizon=1000, external_inputs=())
    and:
        - If `track_states=False`: returns `outputs` (Tensor (batch, horizon, features))
        - If `track_states=True` : returns (outputs, states_history) where
          `states_history` maps `layer_name -> list[state_tensor]`, each
          (batch, horizon, state_dim).
    """
    model_cache = _FORECAST_CACHE.setdefault(model, {})
    key = (track_states, jit_compile)
    if key in model_cache:
        if show_progress:
            print("Using cached forecast function")
        return model_cache[key]

    # capture info outside graph
    input_names = [_input.name for _input in model.inputs]
    external_input_count = len(input_names) - 1
    reservoir_layers = [layer for layer in model.layers if isinstance(layer, BaseReservoir)]

    @tf_function(reduce_retracing=True, jit_compile=jit_compile)
    def _compiled_forecast_fn(
        initial_feedback: tf.Tensor,
        horizon: int = 1000,
        external_inputs: Tuple[tf.Tensor, ...] = (),
    ):
        batch_size, features = _validate_inputs(
            initial_feedback, external_inputs, external_input_count
        )

        horizon_const = tf.convert_to_tensor(horizon, dtype=tf.int32)
        if external_inputs:
            ext_lengths = [tf.shape(ext)[1] for ext in external_inputs]
            horizon_t = tf.minimum(horizon_const, tf.reduce_min(ext_lengths))
        else:
            horizon_t = horizon_const

        t0 = tf.constant(0, tf.int32)
        outputs_ta = tf.TensorArray(
            dtype=model.output.dtype,
            size=horizon_t,
            element_shape=(batch_size, features),
        )

        states_init, state_shape_invariants = (
            _setup_state_arrays(reservoir_layers, horizon_t)
            if track_states
            else ({}, {})
        )

        def cond(t, *_):
            return t < horizon_t

        def body(t, feedback, outputs_ta, states_lv):
            model_inputs = feedback if not external_inputs else [feedback] + [
                tf.expand_dims(ext[:, t, :], axis=1) for ext in external_inputs
            ]
            out_t = model(model_inputs, training=False)
            outputs_ta = outputs_ta.write(t, tf.squeeze(out_t, axis=1))
            new_feedback = tf.reshape(out_t, [batch_size, 1, features])

            if track_states:
                for layer in reservoir_layers:
                    layer_ta_list = states_lv[layer.name]
                    for i, (ta, st) in enumerate(zip(layer_ta_list, layer.get_states())):
                        layer_ta_list[i] = ta.write(t, st)

            return t + 1, new_feedback, outputs_ta, states_lv

        loop_vars = (t0, initial_feedback, outputs_ta, states_init)
        shape_invariants = (
            t0.shape,
            tf.TensorShape([batch_size, 1, features]),
            tf.TensorShape(None),
            state_shape_invariants,
        )

        _, _, outputs_ta, states_out = tf.while_loop(
            cond,
            body,
            loop_vars,
            shape_invariants=shape_invariants,
            parallel_iterations=1,
        )

        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])

        if track_states:
            states_history = {}
            for layer in reservoir_layers:
                stacked = [
                    tf.transpose(ta.stack(), [1, 0, 2]) for ta in states_out[layer.name]
                ]
                states_history[layer.name] = stacked
        else:
            states_history = {}

        return outputs, states_history

    model_cache[key] = _compiled_forecast_fn
    return _compiled_forecast_fn


# -----------------------------------------------------------------------------
# Cache management
# -----------------------------------------------------------------------------
def clear_forecast_cache(model: Optional[tf.keras.Model] = None) -> None:
    """Clear compiled forecast cache (all or for a specific model)."""
    if model is None:
        _FORECAST_CACHE.clear()
    else:
        _FORECAST_CACHE.pop(model, None)


# -----------------------------------------------------------------------------
# Predict factory
# -----------------------------------------------------------------------------
def predict_factory(model: tf.keras.Model) -> Callable:
    """
    Return a compiled prediction function for `model`.

    The returned function has signature:
        predict(data)
    and:
        - `data` is a tensor or list of tensors of shape (batch, time, features)
        - Returns a tensor of shape (batch, time, features)
    """
    if model in _PREDICT_CACHE:
        return _PREDICT_CACHE[model]

    @tf_function(reduce_retracing=True, jit_compile=True)
    def _predict(data: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor, ...]]) -> tf.Tensor:
        """
        Predict the output of the model.
        """
        return model(data, training=False)

    _PREDICT_CACHE[model] = _predict
    return _predict


# -----------------------------------------------------------------------------
# Warmup + Forecast
# -----------------------------------------------------------------------------
def warmup_forecast(
    model: tf.keras.Model,
    warmup_data: Union[tf.Tensor, List[tf.Tensor]],
    horizon: int,
    external_inputs: Union[tf.Tensor, List[tf.Tensor]]=(),
    show_progress: bool = True,
    states: bool = False,
) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
    """
    Warm up the model using ground-truth data, then run an auto-regressive forecast.

    Parameters
    ----------
    model : tf.keras.Model
        The model to forecast.
    warmup_data : Union[tf.Tensor, List[tf.Tensor]]
        The data to warmup the model.
    horizon : int
        The horizon to forecast.
    external_inputs : Union[tf.Tensor, List[tf.Tensor]]
        The external inputs to the model.
    show_progress : bool
        Whether to show progress.
    states : bool
        Whether to track states.

    Returns
    -------
    outputs : tf.Tensor
        - Forecasted data of shape (batch, horizon, features)
    states_history : Dict[str, List[tf.Tensor]]
        - Empty if `states=False`.
        - Maps `layer_name -> list[state_tensor]`, each (batch, horizon, state_dim).
    """
    input_count = len(model.inputs)

    if isinstance(warmup_data, list):
        if len(warmup_data) != input_count:
            raise ValueError(
                f"Expected {input_count} inputs for warmup_data, got {len(warmup_data)}"
            )
        trimmed_warmup_data = [data[:, :-1, :] for data in warmup_data] # for the predict_fn
        initial_feedback = warmup_data[0][:, -1:, :]
    else:
        trimmed_warmup_data = warmup_data[:, :-1, :]
        initial_feedback = warmup_data[:, -1:, :]


    if show_progress:
        print("Warming up model with teacher-forced data…", file=sys.stderr, flush=True)
    _ = predict_factory(model)(trimmed_warmup_data)

    if show_progress:
        print(f"Running auto-regressive forecast for {horizon} steps…", file=sys.stderr, flush=True)
    forecast_fn = forecast_factory(model, track_states=states, jit_compile=True, show_progress=show_progress)

    outputs, states_history = forecast_fn(
        initial_feedback=initial_feedback,
        external_inputs=external_inputs,
        horizon=horizon,
    )

    if show_progress:
        print(f"Forecast completed - output shape {outputs.shape}", file=sys.stderr, flush=True)

    return outputs, states_history


def window_forecast(
    model: tf.keras.Model,
    data: Union[tf.Tensor, List[tf.Tensor]],
    initial_warmup: Union[tf.Tensor, List[tf.Tensor]], # with external inputs or not
    *,
    warmup_time: int,
    forecast_time: int,
    show_progress: bool = True,
    states: bool = False,
) -> Tuple[tf.Tensor, List[Dict[str, List[tf.Tensor]]], tf.Tensor]:
    """
    Run a windowed forecast over the data. The data is split into chunks of size `forecast_time`, and the model is forecasted for each chunk. Thermalizing with some previous real data before each forecast.

    Parameters
    ----------
    model : tf.keras.Model
        The model to forecast.
    data : Union[tf.Tensor, List[tf.Tensor]]
        The data to forecast.
    initial_warmup : Union[tf.Tensor, List[tf.Tensor]]
        The initial warmup data.
    warmup_time : int
        The time to warmup the model.
    forecast_time : int
        The time to forecast.
    show_progress : bool
        Whether to show progress.
    states : bool
        Whether to track states.

    Returns
    -------
    Tuple[tf.Tensor, List[Dict[str, List[tf.Tensor]]], tf.Tensor]
        - Forecasted data of same shape as `data`
        - States, list of length `len(data) / forecast_time` with each element being a dictionary of `layer_name -> list[state_tensor]`, each (batch, forecast_time, state_dim).
        - Prediction starting indices, list of length `len(data) / forecast_time` with each element being the starting index of the forecast for the corresponding chunk.
    """
    if warmup_time <= 0 or forecast_time <= 0:
        raise ValueError("warmup_time and forecast_time must be positive.")

    num_inputs = len(model.inputs)
    if isinstance(data, list):
        if len(data) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {len(data)}.")
        batch, T, features = data[0].shape
        feedback_data = data[0]
        external_inputs = data[1:]
    else:
        if num_inputs != 1:
            raise ValueError(f"Model expects {num_inputs} inputs but `data` is a single tensor.")
        batch, T, features = data.shape
        feedback_data = data
        external_inputs = ()

    # Validate initial_warmup shape minimally
    if isinstance(initial_warmup, list):
        if len(initial_warmup) != num_inputs:
            raise ValueError(f"initial_warmup must have {num_inputs} tensors.")
        initial_external_inputs = initial_warmup[1:]

    else:
        if num_inputs != 1:
            raise ValueError("For multi-input models, pass initial_warmup as a list.")
        initial_external_inputs = ()

    def window_slice(
        x: Union[tf.Tensor, List[tf.Tensor]],
        length: int,
    ) -> List[Union[tf.Tensor, List[tf.Tensor]]]:
        """
        Split tensor(s) along the time dimension into forecast chunks.

        Parameters
        ----------
        x : tf.Tensor or list of tf.Tensor
            Either a tensor of shape (batch, time, features),
            or a list of such tensors.
        length : int
            Length of each forecast chunk.

        Returns
        -------
        list
            If `x` is a tensor → list of tensors [(batch, chunk, features), ...]
            If `x` is a list → list of lists [[t1_chunk, t2_chunk, ...], ...]
            Always of length ceil(time / length).
        """
        if isinstance(x, list):
            tensors = x
            single_input = False
        else:
            tensors = [x]
            single_input = True

        time = tensors[0].shape[1]
        n_chunks = (time + length - 1) // length # ceil(time / length)

        result = []
        for i in range(n_chunks):
            start = i * length
            end = min((i + 1) * length, time)
            chunks = [t[:, start:end, :] for t in tensors]
            result.append(chunks if not single_input else chunks[0])

        return result

    # Pre-slice forecast chunks (still needed for forecast externals)
    feedback_chunks = window_slice(feedback_data, forecast_time)

    has_externals = isinstance(external_inputs, (list, tuple)) and len(external_inputs) > 0
    if has_externals:
        external_chunks = window_slice(list(external_inputs), forecast_time)
    else:
        external_chunks = []

    def externals_forecast_for_chunk(i: int):
        if not has_externals:
            return ()
        return external_chunks[i]

    def feedback_warmup_for_chunk(i: int) -> tf.Tensor:
        """Warmup from the full data timeline: [start-warmup_time, start)."""
        if i == 0:
            return initial_warmup[0] if isinstance(initial_warmup, list) else initial_warmup
        start = i * forecast_time
        s = max(0, start - warmup_time)
        e = start
        fb_warm = feedback_data[:, s:e, :]
        # Fallback to initial_warmup if we don't have at least 1 step
        if fb_warm.shape[1] == 0:
            return initial_warmup[0] if isinstance(initial_warmup, list) else initial_warmup
        return fb_warm

    def externals_warmup_for_chunk(i: int):
        """Warmup externals aligned with the warmup feedback window."""
        if not has_externals:
            return ()
        if i == 0:
            return initial_external_inputs
        start = i * forecast_time
        s = max(0, start - warmup_time)
        e = start
        return [ext[:, s:e, :] for ext in external_inputs]

    outputs_list: List[tf.Tensor] = []
    states_windows: List[Dict[str, List[tf.Tensor]]] = []
    pred_starts: List[int] = []

    n_chunks = len(feedback_chunks)
    for i in range(n_chunks):
        chunk_start = i * forecast_time
        horizon = min(forecast_time, T - chunk_start)
        if horizon <= 0:
            break

        # Build warmup_data with correct arity (1 or multi inputs)
        fb_warm = feedback_warmup_for_chunk(i)
        ex_warm = externals_warmup_for_chunk(i)
        if num_inputs == 1:
            warmup_data = fb_warm
        else:
            # multi-input: teacher-forced warmup **must** include externals
            warmup_data = [fb_warm] + list(ex_warm)

        # Forecast externals aligned with the **current** chunk
        ex_fore = externals_forecast_for_chunk(i)

        out, st_hist = warmup_forecast(
            model=model,
            warmup_data=warmup_data,
            horizon=horizon,
            external_inputs=ex_fore,
            show_progress=show_progress,
            states=states,
        )
        outputs_list.append(out)
        states_windows.append(st_hist)
        pred_starts.append(chunk_start)

    outputs = tf.concat(outputs_list, axis=1)
    # Optional: if you prefer tensor for callers/plots, convert pred_starts:
    # pred_starts = tf.constant(pred_starts, dtype=tf.int32)
    return outputs, states_windows, pred_starts

__all__ = ["forecast_factory", "clear_forecast_cache", "warmup_forecast", "window_forecast"]