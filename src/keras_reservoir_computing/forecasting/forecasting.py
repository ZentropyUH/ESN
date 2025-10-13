import sys
import weakref
from typing import Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.layers.base import BaseReservoir
from keras_reservoir_computing.utils.tensorflow import tf_function

__all__ = ["forecast_factory", "clear_forecast_cache", "warmup_forecast"]

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
    if model in _PREDICT_CACHE:
        return _PREDICT_CACHE[model]

    @tf_function(reduce_retracing=True, jit_compile=True)
    def _predict(data: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor, ...]]) -> tf.Tensor:
        return model(data, training=False)

    _PREDICT_CACHE[model] = _predict
    return _predict


# -----------------------------------------------------------------------------
# Warmup + Forecast
# -----------------------------------------------------------------------------
def warmup_forecast(
    model: tf.keras.Model,
    warmup_data: Union[tf.Tensor, List[tf.Tensor]],
    forecast_data: Union[tf.Tensor, List[tf.Tensor]],
    horizon: int,
    show_progress: bool = True,
    states: bool = False,
) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
    """
    Warm up the model using ground-truth data, then run an auto-regressive forecast.

    Returns
    -------
    outputs : tf.Tensor
        (batch, horizon, features)
    states_history : Dict[str, List[tf.Tensor]]
        Empty if `states=False`.
    """
    input_count = len(model.inputs)

    if isinstance(warmup_data, list) and len(warmup_data) != input_count:
        raise ValueError(
            f"Expected {input_count} inputs for warmup_data, got {len(warmup_data)}"
        )
    if isinstance(forecast_data, list) and len(forecast_data) != input_count:
        raise ValueError(
            f"Expected {input_count} inputs for forecast_data, got {len(forecast_data)}"
        )

    # split forecast_data
    if isinstance(forecast_data, list):
        initial_feedback = forecast_data[0][:, :1, :]
        external_inputs: Tuple[tf.Tensor, ...] = tuple(forecast_data[1:])
    else:
        initial_feedback = forecast_data[:, :1, :]
        external_inputs = ()

    if show_progress:
        print("Warming up model with teacher-forced data…", file=sys.stderr, flush=True)
    predict_fn = predict_factory(model)
    _ = predict_fn(warmup_data)

    if show_progress:
        print(f"Running auto-regressive forecast for {horizon} steps…", file=sys.stderr, flush=True)
    forecast_fn = forecast_factory(model, track_states=states, jit_compile=True)

    outputs, states_history = forecast_fn(
        initial_feedback=initial_feedback,
        external_inputs=external_inputs,
        horizon=horizon,
    )

    if show_progress:
        print(f"Forecast completed - output shape {outputs.shape}", file=sys.stderr, flush=True)

    return outputs, states_history
