"""Reservoir *read-out* training utilities.

This module exposes :class:`ReservoirTrainer`, a thin wrapper that walks
through all :class:`keras_reservoir_computing.layers.readouts.base.ReadOut`
instances contained in a pre-assembled Keras model and trains them **in
correct dependency order**.

The class is intentionally *minimal* - it neither touches nor re-orders model
weights other than the Read-Out layers themselves.  Internally it constructs
*intermediate* sub-models on-demand to supply each Read-Out with the exact
hidden representation it consumes.

Example
-------
>>> trainer = ReservoirTrainer(model, targets, log=True)
>>> trainer.fit_readout_layers(warmup_batch, input_batch)
"""

import gc
import logging
import weakref
from typing import Callable, Dict, List, Union

import tensorflow as tf

from keras_reservoir_computing.layers.readouts.base import ReadOut
from keras_reservoir_computing.utils.tensorflow import suppress_retracing, tf_function

__all__: List[str] = ["ReservoirTrainer"]

logger = logging.getLogger(__name__)

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
        warm_forward_factory._cache = weakref.WeakKeyDictionary()

    if model in warm_forward_factory._cache:
        return warm_forward_factory._cache[model]

    @tf_function(reduce_retracing=True, jit_compile=True)
    def warm_forward(warmup: tf.Tensor, data: tf.Tensor) -> tf.Tensor:
        """Warm-forward the model through the data.

        This function is a wrapper around the model's predict method that sets the
        model's training parameter to False.
        """
        model(warmup)  # warm-up (stateful layers adapt)
        return model(data)  # inference only - no gradients

    warm_forward_factory._cache[model] = warm_forward
    return warm_forward



class ReservoirTrainer:
    """Utility that *sequentially* trains every Read-Out layer in a model.

    The trainer honours the *topological* ordering of ``model.layers`` so that
    Read-Outs that depend on earlier network components are adjusted only
    *after* their prerequisites have converged.

    Parameters
    ----------
    model
        A fully-built Keras model that already contains one or more
        :class:`ReadOut` layers.
    readout_targets
        Mapping ``readout_layer_name -> target_tensor`` providing the ground
        truth for each Read-Out.
    log
        If *True* a concise progress report is emitted via the :pymod:`logging`
        module.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(
        self,
        model: tf.keras.Model,
        readout_targets: Dict[str, tf.Tensor],
        *,
        log: bool = False,
    ) -> None:
        self.model: tf.keras.Model = model
        self.readout_targets: Dict[str, tf.Tensor] = readout_targets
        self.log: bool = log

        if log:
            logging.basicConfig(level=logging.INFO)

        # `model.layers` is already in topological order - filter for ReadOuts
        self.readout_layers_list: List[ReadOut] = [
            layer
            for layer in self.model.layers
            if isinstance(layer, ReadOut) and layer.name in self.readout_targets
        ]

        # Intermediate warm-forwarding functions
        self._warm_forward_functions = weakref.WeakValueDictionary()
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_warm_forward_function(self, layer_name: str) -> Callable:
        """Return (or build) a sub-model that outputs the *input* to a Read-Out.

        The sub-model shares weights with ``self.model`` and therefore incurs
        negligible memory overhead once constructed.

        Parameters
        ----------
        layer_name
            Name of the Read-Out whose input tensor is to be produced.

        Returns
        -------
        tf.keras.Model
            A model mapping ``self.model.input`` to the requested Read-Out
            *input* (not its output!).
        """
        if layer_name not in self._warm_forward_functions:
            layer = next(
                layer_obj
                for layer_obj in self.readout_layers_list
                if layer_obj.name == layer_name
            )
            submodel = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer.input,
            )
            with suppress_retracing():
                self._warm_forward_functions[layer_name] = warm_forward_factory(submodel)
        return self._warm_forward_functions[layer_name]


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_readout_layers(
        self,
        warmup_data: Union[tf.Tensor, List[tf.Tensor]],
        input_data: Union[tf.Tensor, List[tf.Tensor]],
    ) -> None:
        """Train each :class:`ReadOut` layer exactly once in *topological* order.

        The routine iterates over :pyattr:`readout_layers_list`, extracts the
        *correct* hidden representation for that layer via a lazily built
        intermediate model, and calls :pymeth:`ReadOut.fit` with the
        user-supplied target.

        Parameters
        ----------
        warmup_data
            Batch (or list of batches) used solely to *warm* the recurrent
            dynamics before collecting Read-Out inputs.
        input_data
            Actual data from which the Read-Out inputs are computed.

        Notes
        -----
        *Memory usage* - Intermediate tensors are cleared as soon as the
        corresponding Read-Out has been trained to keep the memory footprint
        low when many Read-Outs are present.
        """
        if self.log:
            logger.info("\n=== Training Read-Out layers in topological order ===")

        for readout_layer in self.readout_layers_list:
            layer_name = readout_layer.name
            if self.log:
                logger.info("Processing %s…", layer_name)

            # ----------------------------------------------------------------
            # Build / retrieve sub-model & compute Read-Out input tensor
            # ----------------------------------------------------------------
            warm_forward = self._get_warm_forward_function(layer_name)
            if self.log:
                logger.info("  Warming up + forwarding through %s…", layer_name)

            readout_input = warm_forward(warmup_data, input_data)

            # ----------------------------------------------------------------
            # Fit Read-Out
            # ----------------------------------------------------------------
            target = self.readout_targets[layer_name]
            if self.log:
                logger.info("  Fitting %s…", layer_name)
            readout_layer.fit(readout_input, target)

            # ----------------------------------------------------------------
            # House-keeping - free bulky tensors & intermediate model
            # ----------------------------------------------------------------
            readout_input = None  # hint the GC
            gc.collect()
            # del self._warm_forward_functions[layer_name]
            if self.log:
                logger.info("  %s fitted successfully.", layer_name)

        if self.log:
            logger.info("All Read-Out layers fitted successfully.")
