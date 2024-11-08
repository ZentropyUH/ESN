import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import Layer
from rich.progress import track
import logging

from ..layers import RemoveOutliersAndMean
from ..reservoirs import BaseReservoir
from ..utils import TF_Ridge, timer

logging.basicConfig(level=logging.INFO)

@keras.saving.register_keras_serializable(package="ReservoirComputers", name="ReservoirComputer")
class ReservoirComputer(keras.Model):
    """
    Base Reservoir Computer model.\n
    Works as a base class for different types of Reservoir Computers. The key components will be a reservoir and a readout layer.
    Groups a set of basic functionalities for all Reservoir Computers.

    Args:
        reservoir (keras.Layer): The reservoir layer of the Reservoir Computer.

        readout (keras.Layer): The readout layer of the Reservoir Computer.

        seed (int | None): The seed of the model.
    """
    def __init__(
        self,
        reservoir: BaseReservoir,
        readout: Layer,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reservoir = reservoir
        self.readout = readout
        self.seed = seed

        if seed is None:
            logging.warning("Seed is None. Reproducibility is not guaranteed.")

    def build(self, input_shape):
        """Build the model with the given input shape.

        Args:
            input_shape (tuple): The input shape of the model.
        """
        self.reservoir.build(input_shape)
        self._input_shape = input_shape
        self.reservoir.build(input_shape)
        reservoir_output_shape = self.reservoir.compute_output_shape(input_shape)
        self.readout.build(reservoir_output_shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass of the model.

        Args:
            inputs (Tensor): Input tensor. Of shape (batch_size, timesteps, input_dim).

        Returns:
            Tensor: Output tensor after passing through reservoir and readout layers. Of shape (batch_size, timesteps, output_dim).
        """
        # Pass the inputs through the reservoir
        reservoir_output = self.reservoir(inputs)
        # Pass the reservoir output through the readout
        output = self.readout(reservoir_output)
        return output

    def train(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        train_target: np.ndarray,
        regularization: float,
        log: bool = False
    ) -> float:
        """Train the model with the given data.

        It trains the model fully in one step.

        Args:
            inputs (Tuple[np.array, np.array]): The input data for the training. It is a tuple with the transient data and the training data.
            train_target (np.array): The target data for the training.
            regularization (float): Regularization value for linear readout.

        Returns:
            float: The training loss of the model.
        """
        #unpack the data, this will be a tuple with all inside
        transient_data, train_data = inputs

        # 1. Ensure the model is built
        if not self.built:
            self.build(train_data.shape)

        # 2. Ensure the Echo State Property (ESP) of the model by predicting the transient data
        transient_data = keras.ops.convert_to_tensor(transient_data, dtype="float32")
        with timer("Ensuring ESP", log=log):
            self.ensure_ESP(transient_data)

        # 3. Harvest the reservoir states
        train_data = keras.ops.convert_to_tensor(train_data, dtype="float32")
        with timer("Harvesting reservoir states", log=log):
            harvested_states = self._harvest(train_data)

        # 4. Perform the Ridge regression  and get predictions
        with timer("Calculating readout", log=log):
            predictions = self._calculate_readout(
                harvested_states,
                train_target,
                regularization=regularization
            )

        # Calculate the training loss
        training_loss = self._calculate_loss(predictions, train_target)

        return training_loss

    @tf.function(reduce_retracing=True)
    def ensure_ESP(
        self,
        transient_data: tf.Tensor,
        return_states: bool = False
    ) -> Optional[tf.Tensor]:
        """Ensure the Echo State Property (ESP) of the model by predicting the transient data.

        Args:
            transient_data (tf.Tensor): Transient data to ensure ESP.
        """
        states = self.reservoir.call(transient_data, verbose=0)
        if return_states:
            return states

    def _harvest(
        self,
        train_data: tf.Tensor,
    ) -> tf.Tensor:
        """Harvests the reservoir states after ensuring Echo State Property (ESP).

        This method ensures ESP by predicting the training data and harvests the reservoir states.
        It returns the harvested reservoir states corresponding to the training data.

        Args:
            train_data (tf.Tensor): Data to harvest the reservoir states.

            log (bool): Whether to log the time taken for the operation. Defaults to False.

        Returns:
            tf.Tensor: The harvested reservoir states.
        """
        return self.reservoir.predict(train_data, verbose=0)

    def _calculate_readout(
        self,
        harvested_states: tf.Tensor,
        train_target: np.ndarray,
        regularization: float = 0,
        log: bool = False
    ) -> tf.Tensor:
        """
        Calculate the readout of the model using Ridge regression with SVD solver.

        Args:
            harvested_states (tf.Tensor): The harvested reservoir states.

            train_target (np.ndarray): The target data for the training.

            regularization (float): Regularization value for linear readout.

            log (bool): Whether to log the time taken for the operation. Defaults to False.

        Returns:
            tf.Tensor: The predictions on the training data.
        """
        # Cast the data to float64
        harvested_states = keras.ops.cast(harvested_states, dtype="float64")
        train_target = keras.ops.cast(train_target, dtype="float64")

        regressor = TF_Ridge(alpha=regularization)

        regressor.fit(harvested_states, train_target)

        # Set weights of the readout layer
        coef = regressor._coef
        intercept = regressor._intercept
        self.readout.set_weights([coef, intercept])

        # Get the predictions
        predictions = regressor.predict(harvested_states)

        predictions = tf.expand_dims(predictions, axis=0) # predictions shape is (timesteps, output_dim), we need to add the batch size

        return predictions

    def _calculate_loss(self,
        predictions: tf.Tensor,
        train_target: np.ndarray
    ) -> float:
        # Calculate the loss
        loss = tf.reduce_mean(tf.square(train_target - predictions))

        return loss

    def get_states(self) -> List[tf.Tensor]:
        """Retrieve the current states of all RNN layers in the model.

        Returns:
            List[tf.Tensor]: List of current reservoir states.
        """
        return self.reservoir.get_states()

    def set_states(self, new_states: List[tf.Tensor]) -> None:
        """Set the states of the reservoir.

        Args:
            new_states (List[tf.Tensor]): List of new states to set.
        """
        self.reservoir.set_states(new_states)

    def reset_states(self):
        """Reset internal states of the RNNs of the model."""
        self.reservoir.reset_states()

    def forecast(
        self,
        forecast_length: int,
        forecast_transient_data: np.ndarray,
        val_data: np.ndarray,
        val_target: np.ndarray,
        internal_states: bool = False,
        error_threshold: Optional[float] = None,
    ) -> Tuple[tf.Variable, Optional[tf.Variable], np.ndarray, Optional[int]]:
        """Forecasts the model for a given number of steps.

        It also calculates the cumulative error if the error threshold is specified.
        If `internal_states` is True, it will store the internal states of the reservoir over time.

        Args:
            forecast_length (int): The number of steps to forecast.
            forecast_transient_data (np.ndarray): Transient data to ensure ESP.
            val_data (np.ndarray): Validation data to forecast.
            val_target (np.ndarray): Target data for the validation data.
            internal_states (bool, optional): Whether to store the internal states over time. Defaults to False.
            error_threshold (Optional[float], optional): Threshold for cumulative RMSE. Defaults to None.

        Returns:
            Tuple[tf.Variable, Optional[tf.Variable], np.ndarray, Optional[int]]: A tuple containing predictions, states over time, cumulative error, and steps to exceed error threshold.
        """
        self.reset_states()

        if forecast_length > val_data.shape[1]:
            print("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        predictions, states_over_time = (
            self._initialize_forecast_structures(
                val_data,
                forecast_length,
                internal_states
            )
        )

        cumulative_error, steps_to_exceed_threshold = (
            self._perform_forecasting(
                predictions,
                forecast_transient_data,
                val_target,
                error_threshold,
                states_over_time,
            )
        )

        # Remove the initial placeholder prediction
        predictions = predictions[:, 1:, :]

        return (
            predictions,
            states_over_time,
            cumulative_error,
            steps_to_exceed_threshold,
        )

    def _initialize_forecast_structures(
        self,
        val_data: np.ndarray,
        forecast_length: int,
        internal_states: bool
    ) -> Tuple[tf.Variable, Optional[tf.Variable]]:
        """Initializes structures to store predictions and internal states during forecasting.

        Args:
            val_data (np.ndarray): Validation data.
            forecast_length (int): Number of steps to forecast.
            internal_states (bool): Whether to store internal states.

        Returns:
            Tuple[tf.Variable, Optional[tf.Variable]]: Initialized predictions and states_over_time.
        """
        predictions = tf.Variable(
            tf.zeros((1, forecast_length + 1, val_data.shape[2])), dtype="float32"
        )
        predictions[:, :1, :].assign(
            keras.ops.convert_to_tensor(val_data[:, :1, :], dtype="float32")
        )  # Initialize the first prediction with the first data point


        states_over_time = None
        if internal_states:
            # The idea here is that states can be a list of tensors, for each tensor we will have to accumulate the values over time
            states = self.get_states()

            n_states = len(states)
            features = states[0].shape[1] # Assuming all states have the same number of features

            # We will make states_over_time a tf.Variable with shape (n_states, forecast_length, features)
            states_over_time = tf.Variable(
                tf.zeros((n_states, forecast_length, features)), dtype="float32"
            )

        return (
            predictions,
            states_over_time
        )

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input: tf.Tensor) -> tf.Tensor:
        """Forecasts a single step.

        Args:
            current_input (tf.Tensor): Current input tensor.

        Returns:
            tf.Tensor: Forecasted output tensor.
        """
        return self.call(current_input)

    def _perform_forecasting(
        self,
        predictions: tf.Variable,
        forecast_transient_data: np.ndarray,
        val_target: np.ndarray,
        error_threshold: Optional[float] = None,
        states_over_time: Optional[tf.Variable] = None
    ) -> Tuple[np.ndarray, Optional[int]]:
        """Performs the forecasting process.

        Args:
            predictions (tf.Variable): Predictions array to store forecasted data.
            forecast_transient_data (np.ndarray): Transient data to ensure ESP.
            val_target (np.ndarray): Target data for validation.
            error_threshold (Optional[float]): Threshold for cumulative RMSE.
            states_over_time (Optional[tf.Variable]): Array to store internal states.

        Returns:
            Tuple[np.ndarray, Optional[int]]: Cumulative error array and steps to exceed error threshold.
        """
        steps = predictions.shape[1] - 1

        self.ensure_ESP(forecast_transient_data)

        squared_errors_sum = tf.Variable(0.0, dtype="float32")
        cumulative_error = tf.Variable(tf.zeros((steps), dtype="float32"))
        steps_to_exceed_threshold = None

        for step in track(
            range(steps), description="Forecasting..."
        ):
            current_input = predictions[:, step : step + 1, :]

            # Forecast step
            new_prediction = self.forecast_step(current_input)
            predictions[:, step + 1 : step + 2, :].assign(new_prediction)

            # Store the internal states if required
            if states_over_time is not None:
                states = self.get_states()
                for i, state in enumerate(states):
                    states_over_time[i, step, :].assign(state[0])

            # Analyze if the error threshold is exceeded at the current step
            if error_threshold is not None:
                current_target = val_target[:, step, :]
                current_prediction = predictions[:, step + 1, :]

                # Calculate the squared error
                current_squared_errors = tf.reduce_sum(
                    tf.square(current_target - current_prediction)
                )

                # Update squared errors sum at the current step
                squared_errors_sum.assign_add(current_squared_errors)

                # Calculate cumulative RMSE
                current_rmse = tf.sqrt(
                    squared_errors_sum / keras.ops.cast(step + 1, tf.float32)
                )

                # Store the current RMSE in the cumulative error array
                cumulative_error[step].assign(current_rmse)

                # Check if the current RMSE exceeds threshold
                if current_rmse > error_threshold and steps_to_exceed_threshold is None:
                    steps_to_exceed_threshold = step - 1
                    error_threshold = None

        return cumulative_error.numpy(), steps_to_exceed_threshold

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the model.

        Args:
            input_shape (tuple): The input shape of the model.

        Returns:
            tuple: The output shape of the model.
        """
        reservoir_output_shape = self.reservoir.compute_output_shape(input_shape)
        output_shape = self.readout.compute_output_shape(reservoir_output_shape)
        return output_shape

    @property
    def units(self) -> int:
        return self.reservoir.units

    def get_config(self) -> dict:
        """Returns the configuration of the model.

        Returns:
            dict: Configuration dictionary.
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
        """Creates a model from its configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ReservoirComputerKeras: An instance of the model.
        """
        # Deserialize reservoir and readout layers
        reservoir = keras.layers.deserialize(config.pop('reservoir'))
        readout = keras.layers.deserialize(config.pop('readout'))
        seed = config.pop('seed', None)
        return cls(reservoir=reservoir, readout=readout, seed=seed, **config)

    @classmethod
    def load(cls, path: str) -> "ReservoirComputer":
        """Loads a model from the given path.

        Args:
            path (str): Path to the saved model.

        Returns:
            ReservoirComputerKeras: Loaded model instance.
        """
        model = keras.models.load_model(path, compile=False)
        return model

    def plot(self, **kwargs):
        """Plot the model architecture."""
        if not self.built:
            raise ValueError("Model needs to be built before plotting.")
        input_shape = self.get_build_config()['input_shape']
        input_shape = tuple(input_shape)
        inputs = keras.Input(batch_shape=input_shape, name="Input_Layer")
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return keras.utils.plot_model(model, **kwargs)



@keras.saving.register_keras_serializable(package="ReservoirComputers", name="ReservoirEnsemble")
class ReservoirEnsemble(keras.Model):
    """
    Reservoir Ensemble model.

    Args:
        reservoir_computers (List[ReservoirComputer]): List of Reservoir Computers in the ensemble.
        seed (int | None): The seed of the model.
    """
    def __init__(
        self,
        reservoir_computers: List[ReservoirComputer],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reservoir_computers = reservoir_computers
        self.outlier_removal_layer = RemoveOutliersAndMean(name="outlier_removal")

    def build(self, input_shape):
        """Build the model with the given input shape.

        Args:
            input_shape (tuple): The input shape of the model.
        """
        for reservoir_computer in self.reservoir_computers:
            reservoir_computer.build(input_shape)
        reservoir_computer_output_shape = self.reservoir_computers[0].compute_output_shape(input_shape)
        self.outlier_removal_layer.build(reservoir_computer_output_shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass of the model.

        Args:
            inputs (Tensor): Input tensor. Of shape (batch_size, timesteps, input_dim).

        Returns:
            Tensor: Output tensor after passing through reservoir and readout layers. Of shape (batch_size, timesteps, output_dim).
        """
        # TODO: See if we can use other than a list here.
        reservoir_outputs = [reservoir_computer(inputs) for reservoir_computer in self.reservoir_computers]
        output = self.outlier_removal_layer(reservoir_outputs)

        return output

    def train(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        train_target: np.ndarray,
        regularization: float,
        log: bool = False
    ) -> float:
        """Train the model with the given data.

        It trains the model fully in one step.

        Args:
            inputs (Tuple[np.array, np.array]): The input data for the training. It is a tuple with the transient data and the training data.
            train_target (np.array): The target data for the training.
            regularization (float): Regularization value for linear readout.

        Returns:
            float: The training loss of the model.
        """
        if not self.built:
            self.build(inputs[1].shape)

        losses = []
        for i, reservoir in enumerate(self.reservoir_computers):
            logging.info(f"Training Reservoir {i+1}/{len(self.reservoir_computers)}")
            loss = reservoir.train(inputs, train_target, regularization, log=log)
            losses.append(loss)
        return sum(losses) / len(losses)

    def forecast(
        self,
        forecast_length: int,
        forecast_transient_data: np.ndarray,
        val_data: np.ndarray,
        val_target: np.ndarray,
        internal_states: bool = False,
        error_threshold: Optional[float] = None,
    ) -> Tuple[tf.Variable, Optional[tf.Variable], np.ndarray, Optional[int]]:

        if forecast_length > val_data.shape[1]:
            print("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        # Reset the states of the reservoir computers
        for reservoir_computer in self.reservoir_computers:
            reservoir_computer.reset_states()

        # Initialize the predictions and states_over_time
        predictions, states_over_time = (
            self._initialize_forecast_structures(
                val_data,
                forecast_length,
                internal_states
            )
        )

        # Ensure ESP for all reservoir computers
        for reservoir_computer in self.reservoir_computers:
            reservoir_computer.ensure_ESP(forecast_transient_data)

        # Perform forecasting
        cumulative_error, steps_to_exceed_threshold = self._perform_forecasting(
            predictions,
            val_target,
            error_threshold,
            states_over_time
        )

        # Remove the initial placeholder prediction
        predictions = predictions[:, 1:, :]

        return (
            predictions,
            states_over_time,
            cumulative_error,
            steps_to_exceed_threshold,
        )

    def _initialize_forecast_structures(
        self,
        val_data: np.ndarray,
        forecast_length: int,
        internal_states: bool
    ) -> Tuple[tf.Variable, Optional[tf.Variable]]:
        """Initializes structures to store predictions and internal states during forecasting.

        Args:
            val_data (np.ndarray): Validation data.
            forecast_length (int): Number of steps to forecast.
            internal_states (bool): Whether to store internal states.

        Returns:
            Tuple[tf.Variable, Optional[tf.Variable]]: Initialized predictions and states_over_time.
        """
        # Initialize predictions: shape (batch_size, forecast_length + 1, output_dim)
        predictions = tf.Variable(
            tf.zeros((1, forecast_length + 1, val_data.shape[2])), dtype="float32"
        )
        # Initialize the first prediction with the first data point
        predictions[:, :1, :].assign(
            keras.ops.convert_to_tensor(val_data[:, :1, :], dtype="float32")
        )

        states_over_time = None
        if internal_states:
            # Assuming all reservoir computers have the same number of states and state dimensions
            num_reservoirs = len(self.reservoir_computers)
            states = self.reservoir_computers[0].get_states() # Get the states of the first reservoir, assuming all reservoir computers have the same number of states
            n_states = len(states)
            features = states[0].shape[1] # Assuming all states have the same number of features

            # Initialize states_over_time: shape (num_reservoirs, n_states, forecast_length, features)
            states_over_time = tf.Variable(
                tf.zeros((num_reservoirs, n_states, forecast_length, features)), dtype="float32"
            )

        return (
            predictions,
            states_over_time
        )

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input: tf.Tensor) -> tf.Tensor:
        """Forecasts a single step.

        Args:
            current_input (tf.Tensor): Current input tensor.

        Returns:
            tf.Tensor: Forecasted output tensor.
        """
        return self.call(current_input)

    def _perform_forecasting(
        self,
        predictions: tf.Variable,
        val_target: np.ndarray,
        error_threshold: Optional[float] = None,
        states_over_time: Optional[tf.Variable] = None
    ) -> Tuple[np.ndarray, Optional[int]]:
        """Performs the forecasting process.

        Args:
            predictions (tf.Variable): Predictions array to store forecasted data.
            val_target (np.ndarray): Target data for validation.
            error_threshold (Optional[float]): Threshold for cumulative RMSE.
            states_over_time (Optional[tf.Variable]): Array to store internal states.

        Returns:
            Tuple[np.ndarray, Optional[int]]: Cumulative error array and steps to exceed error threshold.
        """
        steps = predictions.shape[1] - 1

        squared_errors_sum = tf.Variable(0.0, dtype="float32")
        cumulative_error = tf.Variable(tf.zeros((steps), dtype="float32"))
        steps_to_exceed_threshold = None

        for step in track(range(steps), description="Forecasting..."):
            current_input = predictions[:, step : step + 1, :]

            ensemble_output = self.forecast_step(current_input)

            if states_over_time is not None:
                for i, reservoir_computer in enumerate(self.reservoir_computers):
                    states = reservoir_computer.get_states()
                    for j, state in enumerate(states):
                        states_over_time[i, j, step, :].assign(state[0])

            # Store the ensemble output
            predictions[:, step + 1 : step + 2, :].assign(ensemble_output)

            # Update the reservoir computers with the ensemble output
            current_input = ensemble_output

            # Analyze if the error threshold is exceeded at the current step
            if error_threshold is not None:
                current_target = val_target[:, step, :]
                current_prediction = predictions[:, step + 1, :]

                # Calculate the squared error
                current_squared_errors = tf.reduce_sum(
                    tf.square(current_target - current_prediction)
                )

                # Update squared errors sum at the current step
                squared_errors_sum.assign_add(current_squared_errors)

                # Calculate cumulative RMSE
                current_rmse = tf.sqrt(
                    squared_errors_sum / keras.ops.cast(step + 1, tf.float32)
                )

                # Store the current RMSE in the cumulative error array
                cumulative_error[step].assign(current_rmse)

                # Check if the current RMSE exceeds threshold
                if current_rmse > error_threshold and steps_to_exceed_threshold is None:
                    steps_to_exceed_threshold = step - 1
                    error_threshold = None

        return cumulative_error.numpy(), steps_to_exceed_threshold

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the model.

        Args:
            input_shape (tuple): The input shape of the model.

        Returns:
            tuple: The output shape of the model.
        """
        # The output shape will be the same as the input shape
        return input_shape

    def get_states(self) -> List[List[tf.Tensor]]:
        """Retrieve the current states of all RNN layers in the model.

        Returns:
            List[tf.Tensor]: List of current reservoir computers states.
        """
        return [reservoir_computer.get_states() for reservoir_computer in self.reservoir_computers]

    def reset_states(self):
        """Reset internal states of the RNNs of the model."""
        for reservoir_computer in self.reservoir_computers:
            reservoir_computer.reset_states()

    def get_config(self) -> dict:
        """Returns the configuration of the model.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "reservoir_computers": [keras.utils.serialize_keras_object(rc) for rc in self.reservoir_computers],
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "ReservoirEnsemble":
        """Creates a model from its configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ReservoirComputerKeras: An instance of the model.
        """
        # Deserialize reservoir computer and readout layers
        reservoir_computers = [keras.layers.deserialize(reservoir_computer) for reservoir_computer in config.pop('reservoir_computers')]
        return cls(reservoir_computers=reservoir_computers, **config)

    @classmethod
    def load(cls, path: str) -> "ReservoirEnsemble":
        """Loads a model from the given path.

        Args:
            path (str): Path to the saved model.

        Returns:
            ReservoirComputerKeras: Loaded model instance.
        """
        model = keras.models.load_model(path, compile=False)
        return model

    def plot(self, **kwargs):
        """Plot the model architecture."""
        if not self.built:
            raise ValueError("Model needs to be built before plotting.")
        input_shape = self.get_build_config()['input_shape']
        input_shape = tuple(input_shape)
        inputs = keras.Input(batch_shape=input_shape, name="Input_Layer")
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return keras.utils.plot_model(model, **kwargs)
