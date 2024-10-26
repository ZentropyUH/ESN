import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from rich.progress import track
from typing import Any, Tuple, Optional, List

import keras
import tensorflow as tf
from keras import Model


from keras import Layer

from src.customs.custom_reservoirs import BaseReservoir
from src.utils import timer
from src.utils import TF_Ridge
#TODO: Check the shapes of the data, I think now the batch size should not be included in the data
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
        seed: Optional[int] | None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reservoir = reservoir
        self.readout = readout
        self.seed = seed

        if seed is None:
            print("Seed is None. Reproducibility is not guaranteed.")

    def build(self, input_shape):
        """Build the model with the given input shape.

        Args:
            input_shape (tuple): The input shape of the model.
        """
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
        inputs,
        train_target,
        regularization
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
            self.build(transient_data.shape)

        # 2. Ensure the Echo State Property (ESP) of the model by predicting the transient data
        self._ensure_ESP(transient_data, return_states=False)

        # 3. Harvest the reservoir states
        harvested_states = self._harvest(train_data)

        # 4. Perform the Ridge regression  and get predictions
        predictions = self._calculate_readout(
            harvested_states,
            train_target,
            regularization=regularization
        )

        predictions = tf.expand_dims(predictions, axis=0) # predictions shape is (timesteps, output_dim), we need to add the batch size

        # Calculate the training loss
        training_loss = tf.reduce_mean(tf.square(predictions - train_target))

        return training_loss

    def _ensure_ESP(
        self,
        transient_data: np.ndarray,
        return_states: bool = False
    ) -> Optional[list]:
        """Ensure the Echo State Property (ESP) of the model by predicting the transient data.

        Args:
            transient
            return_states (bool, optional): Whether to return the final states of the model. Defaults to False.

        Returns:
            list: List of states of the model.
        """
        with timer("Ensuring ESP"):
            self.reservoir.predict(transient_data)

        if return_states:
            return self.get_states()

    def _harvest(
        self,
        train_data: tf.Tensor
    ) -> tf.Tensor:
        """Harvests the reservoir states after ensuring Echo State Property (ESP).

        This method ensures ESP by predicting the training data and harvests the reservoir states.
        It returns the harvested reservoir states corresponding to the training data.

        Args:
            train_data (tf.Tensor): Data to harvest the reservoir states.

        Returns:
            tf.Tensor: The harvested reservoir states.
        """

        with timer("Harvesting states with combined data"):
            harvested_states = self.reservoir.predict(train_data, verbose=0)

        return harvested_states

    def _calculate_readout(
        self,
        harvested_states: tf.Tensor,
        train_target: np.ndarray,
        regularization: float = 0,
    ) -> float:
        """
        Calculate the readout of the model using Ridge regression with SVD solver.

        Args:
            harvested_states (tf.Tensor): The harvested reservoir states.

            train_target (np.ndarray): The target data for the training.

            regularization (float): Regularization value for linear readout.

        Returns:
            tf.Tensor: The predictions on the training data.
        """
        # Cast the data to float64
        harvested_states = tf.cast(harvested_states, dtype=tf.float64)
        train_target = tf.cast(train_target, dtype=tf.float64)

        regressor = TF_Ridge(alpha=regularization)
        regressor.fit(harvested_states, train_target)

        # Set weights of the readout layer
        coef = regressor._coef
        intercept = regressor._intercept
        self.readout.set_weights([coef, intercept])

        # Get the predictions
        predictions = regressor.predict(harvested_states)

        return predictions

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
    ):
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
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Initializes structures to store predictions and internal states during forecasting.

        Args:
            val_data (np.ndarray): Validation data.
            forecast_length (int): Number of steps to forecast.
            internal_states (bool): Whether to store internal states.

        Returns:
            Tuple[tf.Variable, Optional[tf.Variable]]: Initialized predictions and states_over_time.
        """
        predictions = tf.Variable(
            tf.zeros((1, forecast_length + 1, val_data.shape[2])), dtype=tf.float32
        )
        predictions[:, :1, :].assign(
            tf.convert_to_tensor(val_data[:, :1, :], dtype=tf.float32)
        )  # Initialize the first prediction with the first data point


        states_over_time = None
        if internal_states:
            # The idea here is that states can be a list of tensors, for each tensor we will have to accumulate the values over time
            states = self.get_states()

            n_states = len(states)
            features = states[0].shape[1]

            # We will make states_over_time a tf.Variable with shape (n_states, forecast_length, state_shape)
            states_over_time = tf.Variable(
                tf.zeros((n_states, forecast_length, features)), dtype=tf.float32
            )

        return (
            predictions,
            states_over_time
        )

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

        # self.predict(forecast_transient_data)  # Ensure ESP by predicting transient data
        self._ensure_ESP(forecast_transient_data, return_states=False)

        squared_errors_sum = tf.Variable(0.0, dtype=tf.float32)
        cumulative_error = tf.Variable(tf.zeros((steps), dtype=tf.float32))
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
                    squared_errors_sum / tf.cast(step + 1, tf.float32)
                )

                # Store the current RMSE in the cumulative error array
                cumulative_error[step].assign(current_rmse)

                # Check if the current RMSE exceeds threshold
                if current_rmse > error_threshold and steps_to_exceed_threshold is None:
                    steps_to_exceed_threshold = step - 1
                    error_threshold = None

        return cumulative_error.numpy(), steps_to_exceed_threshold

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input):
        """Forecasts a single step.

        Args:
            current_input (tf.Tensor): Current input tensor.

        Returns:
            tf.Tensor: Forecasted output tensor.
        """
        return self.call(current_input)

    def get_config(self):
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
    def from_config(cls, config):
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
    def load(cls, path):
        """Loads a model from the given path.

        Args:
            path (str): Path to the saved model.

        Returns:
            ReservoirComputerKeras: Loaded model instance.
        """
        model = keras.models.load_model(path, compile=False)
        return model

