import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
from typing import List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import Layer

from keras_reservoir_computing.layers import RemoveOutliersAndMean
from keras_reservoir_computing.reservoirs import BaseReservoir
from keras_reservoir_computing.utils.tf_utils import TF_Ridge
from keras_reservoir_computing.utils.general_utils import timer

logging.basicConfig(level=logging.INFO)


@keras.saving.register_keras_serializable(
    package="ReservoirComputers", name="ReservoirComputer"
)
class ReservoirComputer(keras.Model):
    r"""
    A base reservoir computing model that integrates a reservoir layer and a readout layer.

    This class provides:
    - Forward propagation through a **reservoir** and a **readout** layer.
    - **Training** the readout layer via ridge regression on harvested reservoir states.
    - **Managing reservoir states**, including retrieval, resetting, and modification.
    - **Forecasting** future timesteps using iterative single-step predictions.

    Parameters
    ----------
    reservoir : BaseReservoir
        The reservoir layer (subclass of `BaseReservoir`) responsible for dynamic transformations.
    readout : keras.Layer
        The readout layer (typically a linear layer) applied to the reservoir's output.
    seed : int or None, optional
        Random seed for reproducibility. If None, reproducibility is not guaranteed.
    **kwargs : dict
        Additional keyword arguments passed to the `keras.Model` constructor.

    Attributes
    ----------
    reservoir : BaseReservoir
        The internal reservoir layer handling state updates.
    units : int
        Number of reservoir units.
    readout : keras.Layer
        The internal readout layer used for final predictions.
    seed : int or None
        The random seed for reproducibility.
    _input_shape : tuple of int or None
        Shape of the input data, set during model building.

    Methods
    -------
    **build**(input_shape)
        Builds the model by constructing the reservoir and readout layers.
    **call**(inputs, **kwargs)
        Passes inputs through the reservoir and readout layers.
    **train**(inputs, train_target, regularization, log=False)
        Trains the readout layer using ridge regression on reservoir states.
    **ensure_ESP**(transient_data)
        Ensures the Echo State Property (ESP) by feeding transient data through the reservoir.
    **get_states**()
        Retrieves the current reservoir states.
    **set_states**(new_states)
        Sets the reservoir's internal states.
    **reset_states**()
        Resets all reservoir states to their initial values.
    **forecast**(forecast_length, forecast_transient_data, val_data, store_states=False)
        Forecasts future values using the trained model.
    **forecast_step**(current_input)
        Performs a single-step forecast.
    **compute_output_shape**(input_shape)
        Computes the output shape of the model.
    **get_config**()
        Returns a dictionary containing the model configuration.
    **from_config**(config)
        Creates an instance of the model from a configuration dictionary.
    **load**(path)
        Loads a saved `ReservoirComputer` model from a file.
    **plot**(**kwargs)
        Plots the model architecture.
    **get_build_config**()
        Retrieves build-related information for model reconstruction.
    **build_from_config**(config)
        Marks the model as built based on a configuration.
    """
    def __init__(
        self,
        reservoir: BaseReservoir,
        readout: Layer,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reservoir = reservoir
        self.readout = readout
        self.seed = seed

        if seed is None:
            logging.warning("Seed is None. Reproducibility is not guaranteed.")

    def build(self, input_shape: Tuple[int, ...]) -> None:
        r"""
        Builds the model by first building the reservoir and then the readout layer.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input data, typically (batch_size, timesteps, input_dim).
        """
        if self.built:
            return
        self.reservoir.build(input_shape)
        self._input_shape = input_shape
        reservoir_output_shape = self.reservoir.compute_output_shape(input_shape)
        self.readout.build(reservoir_output_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        r"""
        Forward pass of the reservoir computer, passing the inputs through the reservoir
        and then through the readout layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, timesteps, input_dim).
        **kwargs : dict
            Additional keyword arguments passed to the reservoir's `call`.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, timesteps, output_dim).
        """
        reservoir_output = self.reservoir(inputs, **kwargs)
        output = self.readout(reservoir_output)
        return output

    def train(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        train_target: np.ndarray,
        regularization: float,
        log: bool = False,
    ) -> float:
        r"""
        Trains the reservoir computer model in a single step, using ridge regression
        to fit the readout layer.

        Parameters
        ----------
        inputs : tuple of np.ndarray
            A tuple (transient_data, train_data). Both arrays should be of shape
            (batch_size, timesteps, input_dim). The first is used to ensure ESP,
            the second is used to harvest states and fit the readout.
        train_target : np.ndarray
            The target data to fit. Shape should match the readout's output
            (batch_size, timesteps, output_dim).
        regularization : float
            Ridge regularization parameter.
        log : bool, optional
            Whether to log timing information via the `timer` decorator.

        Returns
        -------
        float
            The training loss (mean squared error over the training set).
        """
        # Unpack
        transient_data, train_data = inputs

        # 1. Ensure the model is built
        if not self.built:
            self.build(train_data.shape)

        # 2. Ensure ESP
        transient_data = keras.ops.convert_to_tensor(transient_data, dtype="float32")
        with timer("Ensuring ESP", log=log):
            self.ensure_ESP(transient_data)

        # 3. Harvest reservoir states from the training data
        train_data = keras.ops.convert_to_tensor(train_data, dtype="float32")
        with timer("Harvesting reservoir states", log=log):
            harvested_states = self._harvest(train_data)

        # 4. Perform ridge regression and get predictions
        with timer("Calculating readout", log=log):
            predictions = self._calculate_readout(
                harvested_states, train_target, regularization=regularization
            )

        # 5. Calculate the training loss
        training_loss = self._calculate_loss(predictions, train_target)
        return float(training_loss)

    @tf.function(autograph=False)
    def ensure_ESP(self, transient_data: tf.Tensor) -> tf.Tensor:
        r"""
        Ensures the Echo State Property (ESP) by making a forward pass through
        the reservoir with transient data.

        Parameters
        ----------
        transient_data : tf.Tensor
            Transient data to wash out the reservoir states,
            shape (batch_size, timesteps, input_dim).

        Returns
        -------
        tf.Tensor
            Reservoir states or outputs corresponding to the transient data.
            Shape depends on the reservoir architecture.
        """
        states = self.reservoir.call(inputs=transient_data)
        return states

    # @tf.function(autograph=False)
    def _harvest(self, train_data: tf.Tensor) -> tf.Tensor:
        r"""
        Harvests the reservoir states by calling the reservoir in inference mode
        (via `predict`) on the training data.

        Parameters
        ----------
        train_data : tf.Tensor
            Training data of shape (batch_size, timesteps, input_dim).

        Returns
        -------
        tf.Tensor
            The harvested reservoir states, typically (batch_size, timesteps, reservoir_units).
        """
        return self.reservoir.predict(train_data, verbose=0)

    def _calculate_readout(
        self,
        harvested_states: tf.Tensor,
        train_target: np.ndarray,
        regularization: float = 0,
        log: bool = False,
    ) -> tf.Tensor:
        r"""
        Fits a ridge regression model (via `TF_Ridge`) to map harvested_states
        to the train_target. Then updates the readout layer weights from the
        fitted model. Finally, it returns the predictions on the training set.

        Parameters
        ----------
        harvested_states : tf.Tensor
            Reservoir states of shape (batch_size, timesteps, reservoir_units).
        train_target : np.ndarray
            Target data of shape (batch_size, timesteps, output_dim).
        regularization : float, optional
            Regularization (alpha) for ridge regression.
        log : bool, optional
            If True, logs timing information.
            Currently unused inside this method but kept for interface consistency.

        Returns
        -------
        tf.Tensor
            Predictions on the training data, expanded to shape
            (1, timesteps, output_dim) to mimic a batch dimension of 1.
        """
        # Cast to float64 for SVD-based ridge solver
        harvested_states = keras.ops.cast(harvested_states, dtype="float64")
        train_target = keras.ops.cast(train_target, dtype="float64")

        regressor = TF_Ridge(alpha=regularization)
        regressor.fit(harvested_states, train_target)

        coef = regressor._coef  # shape (reservoir_units, output_dim)
        intercept = regressor._intercept  # shape (output_dim,)
        # Update readout weights
        self.readout.set_weights([coef, intercept])

        # Predict on the harvested states
        predictions = regressor.predict(harvested_states)  # (timesteps, output_dim)
        predictions = tf.expand_dims(
            predictions, axis=0
        )  # => (1, timesteps, output_dim)

        return predictions

    def _calculate_loss(
        self, predictions: tf.Tensor, train_target: np.ndarray
    ) -> tf.Tensor:
        r"""
        Computes mean squared error (MSE) between predictions and the target.

        Parameters
        ----------
        predictions : tf.Tensor
            Predictions of shape (1, timesteps, output_dim).
        train_target : np.ndarray
            Ground truth target of shape (batch_size, timesteps, output_dim).
            Typically, batch_size==1 in this approach, but not strictly enforced.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the MSE loss.
        """
        loss = tf.reduce_mean(tf.square(train_target - predictions))
        return loss

    def get_states(self) -> List[tf.Tensor]:
        r"""
        Retrieves the current internal states from the reservoir.

        Returns
        -------
        List of tf.Tensor
            A list containing each state's tensor (e.g., hidden state).
        """
        return self.reservoir.get_states()

    def set_states(self, new_states: List[tf.Tensor]) -> None:
        r"""
        Sets the reservoir's internal states to the provided tensors.

        Parameters
        ----------
        new_states : list of tf.Tensor
            The new reservoir states to assign.
        """
        self.reservoir.set_states(new_states)

    def reset_states(self) -> None:
        r"""
        Resets the reservoir's internal states to their default (typically zeros).
        """
        self.reservoir.reset_states()

    def forecast(
        self,
        forecast_length: int,
        forecast_transient_data: np.ndarray,
        val_data: np.ndarray,
        store_states: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        r"""
        Generates a forecast of a given length, using single-step predictions in a loop.

        Parameters
        ----------
        forecast_length : int
            Number of timesteps to forecast.
        forecast_transient_data : np.ndarray
            Data used to wash out states before starting the forecast,
            shape (batch_size, transient_steps, input_dim).
        val_data : np.ndarray
            Validation data to serve as the initial input for forecasting.
            Must have shape (batch_size, timesteps, input_dim) with
            timesteps >= 1 to extract the initial point.
        store_states : bool, optional
            If True, returns a second output capturing internal states at each step.
            Otherwise, returns None for the second output.

        Returns
        -------
        tuple
            (predictions_out, states_out)
            - predictions_out : tf.Tensor
                Forecasted output, shape (batch_size, forecast_length, output_dim).
            - states_out : tf.Tensor or None
                If `store_states=True`, shape depends on the reservoir; otherwise None.
        """
        self.reset_states()

        if forecast_length > val_data.shape[1]:
            print("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        initial_point = val_data[:, :1, :]  # (batch_size, 1, input_dim)

        # Convert to Tensors
        forecast_transient_data_tf = tf.convert_to_tensor(
            forecast_transient_data, dtype=tf.float32
        )
        initial_point_tf = tf.convert_to_tensor(initial_point, dtype=tf.float32)

        # Run the loop
        predictions_out, states_out = self._perform_forecasting_fast_with_states(
            initial_point_tf, forecast_transient_data_tf, steps=forecast_length
        )
        # Remove the initial step from predictions
        predictions_out = predictions_out[:, 1:, :]

        if store_states:
            return predictions_out, states_out
        return predictions_out, None

    @tf.function
    def _perform_forecasting_fast_with_states(
        self,
        initial_point: tf.Tensor,
        forecast_transient_data: tf.Tensor,
        steps: int,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        r"""
        Internal helper to perform multi-step forecasting in a loop,
        optionally capturing the states at each step.

        Steps:
        1) Ensure ESP with transient data.
        2) Store the initial point in a TensorArray.
        3) Iteratively forecast steps, storing predictions and states.

        Parameters
        ----------
        initial_point : tf.Tensor
            Shape (batch_size, 1, input_dim), the first input for forecasting.
        forecast_transient_data : tf.Tensor
            Shape (batch_size, transient_timesteps, input_dim), used to wash out states.
        steps : int
            Number of timesteps to forecast.

        Returns
        -------
        tuple
            (predictions_out, states_out_list)
            - predictions_out : tf.Tensor
                Shape (batch_size, steps+1, input_dim) containing the forecast,
                including the initial point at index 0.
            - states_out_list : list of tf.Tensor
                Each tensor captures the states across forecasting steps.
                states_out_list[i] has shape (batch_size, steps, units_i).
        """
        # 1) Ensure ESP
        _ = self.ensure_ESP(forecast_transient_data)

        # 2) Prepare a TensorArray for predictions
        predictions_ta = tf.TensorArray(
            dtype=tf.float32,
            size=steps + 1,
            element_shape=(None, 1, None),
        )
        predictions_ta = predictions_ta.write(0, initial_point)

        # 3) Prepare TAs for states
        init_states = self.get_states()  # list of (batch_size, units_i)
        states_ta = [
            tf.TensorArray(
                dtype=tf.float32,
                size=steps,
                element_shape=s.shape,
            )
            for s in init_states
        ]

        # Loop condition
        def loop_cond(step, preds_ta, st_ta):
            return step < steps

        # Loop body
        def loop_body(step, preds_ta, st_ta):
            current_input = preds_ta.read(step)  # (batch_size, 1, input_dim)
            new_prediction = self.forecast_step(current_input)
            preds_ta = preds_ta.write(step + 1, new_prediction)

            new_states = self.get_states()
            for i, state_tensor in enumerate(new_states):
                st_ta[i] = st_ta[i].write(step, state_tensor)

            return step + 1, preds_ta, st_ta

        step_init = tf.constant(0, dtype=tf.int32)
        loop_vars = (step_init, predictions_ta, states_ta)
        _, final_predictions_ta, final_states_ta = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=loop_vars,
            parallel_iterations=1,
        )

        # Convert predictions to final tensor => (steps+1, batch_size, 1, input_dim)
        preds_stacked = (
            final_predictions_ta.stack()
        )  # => shape (steps+1, batch_size, 1, input_dim)
        preds_stacked = tf.squeeze(
            preds_stacked, axis=2
        )  # => (steps+1, batch_size, input_dim)
        predictions_out = tf.transpose(
            preds_stacked, perm=[1, 0, 2]
        )  # => (batch_size, steps+1, input_dim)
        predictions_out = tf.stop_gradient(predictions_out)

        # Convert states to final list => each (batch_size, steps, units_i)
        states_out_list = []
        for ta in final_states_ta:
            st_stacked = ta.stack()  # (steps, batch_size, units_i)
            st_stacked = tf.transpose(st_stacked, perm=[1, 0, 2])
            st_stacked = tf.stop_gradient(st_stacked)
            states_out_list.append(st_stacked)

        return predictions_out, states_out_list

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input: tf.Tensor) -> tf.Tensor:
        r"""
        Forecasts a single step by calling the model on the current input.

        Parameters
        ----------
        current_input : tf.Tensor
            A tensor of shape (batch_size, 1, input_dim).

        Returns
        -------
        tf.Tensor
            Single-step forecast, shape (batch_size, 1, output_dim).
        """
        return self.call(current_input)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        r"""
        Computes the output shape by chaining reservoir and readout shapes.

        Parameters
        ----------
        input_shape : tuple of int
            The shape of the input (batch_size, timesteps, input_dim).

        Returns
        -------
        tuple of int
            The computed output shape (batch_size, timesteps, output_dim).
        """
        reservoir_output_shape = self.reservoir.compute_output_shape(input_shape)
        output_shape = self.readout.compute_output_shape(reservoir_output_shape)
        return tuple(output_shape)

    @property
    def units(self) -> int:
        r"""
        Number of reservoir units.

        Returns
        -------
        int
            The number of units in the reservoir layer.
        """
        return self.reservoir.units

    def get_config(self) -> dict:
        r"""
        Returns a Python dictionary containing the configuration
        used to initialize this `ReservoirComputer`.

        Returns
        -------
        dict
            A dictionary containing:
            - reservoir : serialized reservoir layer
            - readout : serialized readout layer
            - seed : the random seed
        """
        config = super().get_config()
        config.update(
            {
                "reservoir": keras.utils.serialize_keras_object(self.reservoir),
                "readout": keras.utils.serialize_keras_object(self.readout),
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "ReservoirComputer":
        r"""
        Creates a `ReservoirComputer` from its configuration.

        Parameters
        ----------
        config : dict
            A configuration dictionary (as returned by `get_config`).

        Returns
        -------
        ReservoirComputer
            The instantiated ReservoirComputer.
        """
        reservoir = keras.layers.deserialize(config.pop("reservoir"))
        readout = keras.layers.deserialize(config.pop("readout"))
        seed = config.pop("seed", None)
        return cls(reservoir=reservoir, readout=readout, seed=seed, **config)

    @classmethod
    def load(cls, path: str) -> "ReservoirComputer":
        r"""
        Loads a `ReservoirComputer` from a given file path.

        Parameters
        ----------
        path : str
            Path to the saved Keras model.

        Returns
        -------
        ReservoirComputer
            The loaded model.
        """
        model = keras.models.load_model(path, compile=False)
        return model

    def plot(self, **kwargs):
        r"""
        Plots the model architecture using Keras's `plot_model` utility.

        Raises
        ------
        ValueError
            If the model has not been built yet.

        Returns
        -------
        keras.utils.vis_utils.Dot
            The plot of the model.
        """
        if not self.built:
            raise ValueError("Model needs to be built before plotting.")
        input_shape = self.get_build_config()["input_shape"]
        input_shape = tuple(input_shape)
        inputs = keras.Input(batch_shape=input_shape, name="Input_Layer")
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return keras.utils.plot_model(model, **kwargs)

    def get_build_config(self) -> dict:
        r"""
        Returns a dictionary with information needed to rebuild this model
        or to visualize it (e.g. in `plot()`).

        Returns
        -------
        dict
            A dictionary containing "input_shape" if the model has been built,
            otherwise an empty dict.
        """
        return (
            {"input_shape": self._input_shape}
            if getattr(self, "_input_shape", None) is not None
            else {}
        )

    def build_from_config(self, config: dict) -> None:
        r"""
        Builds the model from a given configuration.

        Parameters
        ----------
        config : dict
            A configuration dictionary containing building parameters.

        Notes
        -----
        Currently sets `self.built = True` as a placeholder.
        """
        self.built = True  # This is a manual override/patch.


@keras.saving.register_keras_serializable(
    package="ReservoirComputers", name="ReservoirEnsemble"
)
class ReservoirEnsemble(keras.Model):
    r"""
    An ensemble of multiple `ReservoirComputer` models whose outputs are combined
    via an outlier removal layer (`RemoveOutliersAndMean`).

    This ensemble:
    - Builds each `ReservoirComputer` on the same input shape.
    - Stacks their outputs to shape (num_reservoirs, batch, timesteps, output_dim).
    - Applies `RemoveOutliersAndMean` to remove outliers across ensemble members
      and then returns the mean of the inlier outputs.

    Parameters
    ----------
    reservoir_computers : list of ReservoirComputer
        The list of reservoir computers forming the ensemble.
    seed : int or None, optional
        Random seed for reproducibility. If None, reproducibility is not guaranteed.
    **kwargs : dict
        Additional keyword arguments for the `keras.Model` constructor.
    """

    def __init__(
        self, reservoir_computers: List[ReservoirComputer], seed=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.reservoir_computers = reservoir_computers
        self.outlier_removal_layer = RemoveOutliersAndMean(name="outlier_removal")
        self.seed = seed

        if seed is None:
            logging.warning("Seed is None. Reproducibility is not guaranteed.")

    def build(self, input_shape: Tuple[int, ...]) -> None:
        r"""
        Builds all reservoir computers in the ensemble and the outlier-removal layer.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input data (batch_size, timesteps, input_dim).
        """
        if self.built:
            return
        for reservoir_computer in self.reservoir_computers:
            reservoir_computer.build(input_shape)

        reservoir_computer_output_shape = self.reservoir_computers[
            0
        ].compute_output_shape(input_shape)

        self.outlier_removal_layer.build(reservoir_computer_output_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        r"""
        Forwards the input through each `ReservoirComputer`, stacks the outputs,
        then applies the outlier-removal layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, timesteps, input_dim).
        **kwargs : dict
            Additional keyword arguments passed to each reservoir_computer's call.

        Returns
        -------
        tf.Tensor
            The ensemble-averaged (after outlier removal) output,
            shape (batch_size, timesteps, output_dim).
        """
        reservoir_outputs = tf.stack(
            [
                reservoir_computer(inputs)
                for reservoir_computer in self.reservoir_computers
            ]
        )  # shape => (num_reservoirs, batch_size, timesteps, output_dim)
        output = self.outlier_removal_layer(reservoir_outputs)
        return output

    def train(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        train_target: np.ndarray,
        regularization: float,
        log: bool = False,
    ) -> float:
        r"""
        Trains each `ReservoirComputer` in the ensemble.

        Parameters
        ----------
        inputs : tuple of np.ndarray
            A tuple (transient_data, train_data), each shaped (batch_size, timesteps, input_dim).
        train_target : np.ndarray
            The target data for fitting, shape (batch_size, timesteps, output_dim).
        regularization : float
            Ridge regularization parameter.
        log : bool, optional
            Whether to log timing information (passed to each RC).

        Returns
        -------
        float
            The average training loss across all reservoir computers in the ensemble.
        """
        if not self.built:
            self.build(inputs[1].shape)

        losses = []
        for i, reservoir in enumerate(self.reservoir_computers):
            logging.info(f"Training Reservoir {i+1}/{len(self.reservoir_computers)}")
            loss = reservoir.train(inputs, train_target, regularization, log=log)
            losses.append(loss)
        return float(sum(losses) / len(losses))

    @tf.function
    def ensure_ESP(self, transient_data: tf.Tensor) -> None:
        r"""
        Ensures ESP in each `ReservoirComputer` by calling their `ensure_ESP` method.

        Parameters
        ----------
        transient_data : tf.Tensor
            Data used to wash out states, shape (batch_size, timesteps, input_dim).
        """
        for rc in self.reservoir_computers:
            rc.ensure_ESP(transient_data)

    def forecast(
        self,
        forecast_length: int,
        forecast_transient_data: np.ndarray,
        val_data: np.ndarray,
        store_states: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        r"""
        Forecasts the ensemble for a given number of steps, following a similar procedure
        to a single `ReservoirComputer` but stacking each computer's outputs and removing
        outliers.

        Parameters
        ----------
        forecast_length : int
            Number of timesteps to forecast.
        forecast_transient_data : np.ndarray
            Data used to ensure ESP, shape (batch_size, transient_timesteps, input_dim).
        val_data : np.ndarray
            Validation data from which to start forecasting,
            shape (batch_size, timesteps, input_dim).
        store_states : bool, optional
            If True, returns a second output capturing states across timesteps; else None.

        Returns
        -------
        tuple
            (predictions_out, states_out)
            - predictions_out : tf.Tensor
                Forecasted output, shape (batch_size, forecast_length, output_dim).
            - states_out : None or nested list of tf.Tensor
                If `store_states=True`, shape(s) depend on each reservoir's internal states.
        """
        self.reset_states()

        if forecast_length > val_data.shape[1]:
            logging.info("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        initial_point = val_data[:, :1, :]  # (batch_size,1,input_dim)
        forecast_transient_data_tf = tf.convert_to_tensor(
            forecast_transient_data, dtype=tf.float32
        )
        initial_point_tf = tf.convert_to_tensor(initial_point, dtype=tf.float32)

        predictions_out, states_out = self._perform_forecasting_fast_with_states(
            initial_point_tf,
            forecast_transient_data_tf,
            steps=forecast_length,
            store_states=store_states,
        )
        # Remove the initial step => shape => (batch_size, forecast_length, output_dim)
        predictions_out = predictions_out[:, 1:, :]

        if store_states:
            return predictions_out, states_out
        return predictions_out, None

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input: tf.Tensor) -> tf.Tensor:
        r"""
        Single-step forecast by calling `self.call` on the current input.

        Parameters
        ----------
        current_input : tf.Tensor
            Shape (batch_size, 1, input_dim).

        Returns
        -------
        tf.Tensor
            Forecasted output, shape (batch_size, 1, output_dim).
        """
        return self.call(current_input)

    @tf.function
    def _perform_forecasting_fast_with_states(
        self,
        initial_point: tf.Tensor,
        forecast_transient_data: tf.Tensor,
        steps: int,
        store_states: bool = False,
    ) -> Tuple[tf.Tensor, Optional[List[List[tf.Tensor]]]]:
        r"""
        Internal multi-step forecast using tf.while_loop, stacking the ensemble outputs,
        and optionally storing each internal state from each `ReservoirComputer`.

        Parameters
        ----------
        initial_point : tf.Tensor
            Shape (batch_size, 1, input_dim). The initial input for forecasting.
        forecast_transient_data : tf.Tensor
            Shape (batch_size, transient_timesteps, input_dim). Used to wash out states.
        steps : int
            Number of forecasting steps.
        store_states : bool, optional
            Whether to store the states from each reservoir computer at each step.

        Returns
        -------
        tuple
            (predictions_out, states_out_list)
            - predictions_out : tf.Tensor
                Shape (batch_size, steps+1, output_dim).
            - states_out_list : None or list of lists of tf.Tensor
                If `store_states=True`, for each ReservoirComputer and each of its states,
                a Tensor of shape (batch_size, steps, units_i).
        """
        # 1) Ensure ESP in each reservoir
        self.ensure_ESP(forecast_transient_data)

        # 2) TensorArray for predictions
        predictions_ta = tf.TensorArray(
            dtype=tf.float32,
            size=steps + 1,
            element_shape=(None, 1, None),
        )
        predictions_ta = predictions_ta.write(0, initial_point)

        # 3) If storing states, build a TensorArray for each reservoir's states
        states_ta = None
        if store_states:
            states_ta = [
                [
                    tf.TensorArray(
                        dtype=tf.float32,
                        size=steps,
                        element_shape=state.shape,
                    )
                    for state in rc.get_states()
                ]
                for rc in self.reservoir_computers
            ]

        def loop_cond(step, preds_ta, st_ta):
            return step < steps

        def loop_body(step, preds_ta, st_ta):
            current_input = preds_ta.read(step)
            new_prediction = self.forecast_step(current_input)
            preds_ta = preds_ta.write(step + 1, new_prediction)

            if store_states:
                new_states_nested = [rc.get_states() for rc in self.reservoir_computers]
                for rc_idx, rc_states in enumerate(new_states_nested):
                    for state_idx, st_tensor in enumerate(rc_states):
                        st_ta[rc_idx][state_idx] = st_ta[rc_idx][state_idx].write(
                            step, st_tensor
                        )
            return step + 1, preds_ta, st_ta

        step_init = tf.constant(0, dtype=tf.int32)
        loop_vars = (step_init, predictions_ta, states_ta)
        _, final_predictions_ta, final_states_ta = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=loop_vars,
            parallel_iterations=1,
        )

        # Convert predictions => (steps+1, batch_size, 1, output_dim)
        preds_stacked = (
            final_predictions_ta.stack()
        )  # => (steps+1, batch_size, 1, output_dim)
        preds_stacked = tf.squeeze(
            preds_stacked, axis=2
        )  # => (steps+1, batch_size, output_dim)
        predictions_out = tf.transpose(
            preds_stacked, [1, 0, 2]
        )  # => (batch_size, steps+1, output_dim)
        predictions_out = tf.stop_gradient(predictions_out)

        if store_states:
            states_out_list = []
            for rc_tas in final_states_ta:  # each reservoir's list
                rc_states = []
                for ta in rc_tas:
                    st_stacked = ta.stack()  # => (steps, batch_size, units)
                    st_stacked = tf.transpose(
                        st_stacked, perm=[1, 0, 2]
                    )  # => (batch_size, steps, units)
                    st_stacked = tf.stop_gradient(st_stacked)
                    rc_states.append(st_stacked)
                states_out_list.append(rc_states)
            return predictions_out, states_out_list

        return predictions_out, None

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        r"""
        Computes the output shape of the ensemble. In this implementation,
        the output shape is the same as the input shape (batch_size, timesteps, input_dim)
        or adjusted according to the final outlier-removal layer.

        Parameters
        ----------
        input_shape : tuple of int
            The input shape (batch_size, timesteps, input_dim).

        Returns
        -------
        tuple of int
            The computed output shape.
        """
        # If needed, you can refine this to check the shape from outlier_removal_layer.
        return input_shape

    def get_states(self) -> List[List[tf.Tensor]]:
        r"""
        Retrieves the internal states from each `ReservoirComputer` in the ensemble.

        Returns
        -------
        list of list of tf.Tensor
            A nested list, where each sub-list corresponds to one `ReservoirComputer`
            and contains its internal state tensors.
        """
        return [
            reservoir_computer.get_states()
            for reservoir_computer in self.reservoir_computers
        ]

    def reset_states(self) -> None:
        r"""
        Resets the internal states for every `ReservoirComputer` in the ensemble.
        """
        for reservoir_computer in self.reservoir_computers:
            reservoir_computer.reset_states()

    def get_config(self) -> dict:
        r"""
        Returns a configuration dictionary that describes how to re-instantiate
        this ensemble model.

        Returns
        -------
        dict
            A dictionary containing:
            - reservoir_computers : list of serialized `ReservoirComputer` objects
        """
        config = super().get_config()
        config.update(
            {
                "reservoir_computers": [
                    keras.utils.serialize_keras_object(rc)
                    for rc in self.reservoir_computers
                ],
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "ReservoirEnsemble":
        r"""
        Creates a `ReservoirEnsemble` from its configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        ReservoirEnsemble
            An instance of the ensemble model.
        """
        reservoir_computers = [
            keras.layers.deserialize(reservoir_computer)
            for reservoir_computer in config.pop("reservoir_computers")
        ]
        return cls(reservoir_computers=reservoir_computers, **config)

    @classmethod
    def load(cls, path: str) -> "ReservoirEnsemble":
        r"""
        Loads a `ReservoirEnsemble` from a saved model file.

        Parameters
        ----------
        path : str
            The path to the saved Keras model.

        Returns
        -------
        ReservoirEnsemble
            The loaded ensemble model.
        """
        model = keras.models.load_model(path, compile=False)
        return model

    def plot(self, **kwargs):
        r"""
        Plots the architecture of the ensemble using Keras's `plot_model` utility.

        Raises
        ------
        ValueError
            If the model is not built yet.
        """
        if not self.built:
            raise ValueError("Model needs to be built before plotting.")
        input_shape = self.get_build_config()["input_shape"]
        input_shape = tuple(input_shape)
        inputs = keras.Input(batch_shape=input_shape, name="Input_Layer")
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return keras.utils.plot_model(model, **kwargs)

    def get_build_config(self) -> dict:
        r"""
        Returns model build information for plotting or re-initializing.

        Returns
        -------
        dict
            A dictionary containing "input_shape" if set; otherwise empty.
        """
        return (
            {"input_shape": self._input_shape}
            if getattr(self, "_input_shape", None) is not None
            else {}
        )

    def build_from_config(self, config: dict) -> None:
        r"""
        Builds the ensemble from a given configuration.

        Parameters
        ----------
        config : dict
            Build configuration dictionary.

        Notes
        -----
        Currently sets `self.built = True` as a placeholder.
        """
        self.built = True
