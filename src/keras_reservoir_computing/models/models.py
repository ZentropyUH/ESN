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


@keras.saving.register_keras_serializable(
    package="ReservoirComputers", name="ReservoirComputer"
)
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
        **kwargs,
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
        log: bool = False,
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
        # unpack the data, this will be a tuple with all inside
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
                harvested_states, train_target, regularization=regularization
            )

        # Calculate the training loss
        training_loss = self._calculate_loss(predictions, train_target)

        return training_loss

    @tf.function(autograph=False)
    def ensure_ESP(self, transient_data: tf.Tensor) -> tf.Tensor:
        """Ensure the Echo State Property (ESP) of the model by predicting the transient data.

        Will ensure the ESP of the model by predicting the transient data. It returns the reservoir states corresponding to the transient data, used for the ESP index method.

        Args:
            transient_data (tf.Tensor): Transient data to ensure ESP.

        Returns:
            tf.Tensor: The reservoir states corresponding to the transient data
        """
        states = self.reservoir.call(inputs=transient_data, verbose=0)

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

        Returns:
            tf.Tensor: The harvested reservoir states.
        """
        return self.reservoir.predict(train_data, verbose=0)

    def _calculate_readout(
        self,
        harvested_states: tf.Tensor,
        train_target: np.ndarray,
        regularization: float = 0,
        log: bool = False,
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

        predictions = tf.expand_dims(
            predictions, axis=0
        )  # predictions shape is (timesteps, output_dim), we need to add the batch size

        return predictions

    def _calculate_loss(
        self, predictions: tf.Tensor, train_target: np.ndarray
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
        store_states: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Forecasts the model for a given number of steps, optionally storing internal states.

        Args:
            forecast_length (int): Number of steps to forecast.
            forecast_transient_data (np.ndarray): Transient data to ensure ESP.
            val_data (np.ndarray): Validation data to forecast from.
            store_states (bool): Whether to store internal states for each forecast step.

        Returns:
            (predictions, states_out):
            - predictions: tf.Tensor of shape (1, forecast_length, output_dim)
            - states_out: tf.Tensor of shape (n_states, forecast_length, units) if store_states=True, else None
        """
        self.reset_states()
        if forecast_length > val_data.shape[1]:
            print("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        # Take the first time step as the initial point
        initial_point = val_data[:, :1, :]  # shape (1,1,input_dim)

        # Convert to Tensor
        forecast_transient_data_tf = tf.convert_to_tensor(
            forecast_transient_data, dtype=tf.float32
        )
        initial_point_tf = tf.convert_to_tensor(initial_point, dtype=tf.float32)

        # Run the loop, getting predictions plus optional states
        predictions_out, states_out = self._perform_forecasting_fast_with_states(
            initial_point_tf,
            forecast_transient_data_tf,
            steps=forecast_length,
        )

        # The returned predictions contain the initial point in index 0; remove it
        # predictions_out shape: (1, steps+1, input_dim)
        predictions_out = predictions_out[:, 1:, :]

        if store_states:
            return predictions_out, states_out
        return predictions_out, None

    @tf.function(autograph=False)
    def _perform_forecasting_fast_with_states(
        self,
        initial_point: tf.Tensor,
        forecast_transient_data: tf.Tensor,
        steps: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs the forecasting process in TensorFlow graph mode to reduce Python overhead. It also stores the internal states.

        Args:
            initial_point: shape (1, 1, input_dim)
            forecast_transient_data: shape (transient_length, 1, input_dim)
            steps: Number of steps to forecast.

        Returns:
            predictions_out: shape (1, steps+1, input_dim)
            states_out: shape (n_states, steps, units)
        """
        # Ensure ESP. We don't need the returned states
        _ = self.ensure_ESP(forecast_transient_data)

        # Prepare the predictions array
        predictions_ta = tf.TensorArray(
            dtype=tf.float32,
            size=steps + 1,
            element_shape=(1, 1, initial_point.shape[2]),
        )
        predictions_ta = predictions_ta.write(0, initial_point)

        # Prepare the states array
        current_states = self.get_states()  # list of shape [1, units]
        n_states = len(current_states)
        units = current_states[0].shape[1]
        states_ta = tf.TensorArray(
            dtype=tf.float32, size=steps, element_shape=(n_states, units)
        )

        def loop_cond(step, preds_ta, st_ta):
            return step < steps

        def loop_body(step, preds_ta, st_ta):
            current_input = preds_ta.read(step)  # shape (1,1,input_dim)
            new_prediction = self.forecast_step(current_input)  # shape (1,1,input_dim)

            # Write new prediction
            preds_ta = preds_ta.write(step + 1, new_prediction)

            # Write new states
            st_list = self.get_states()  # each is (1, units)
            st_concat = tf.concat(
                [s[0] for s in st_list], axis=0
            )  # shape (n_states*units,)
            st_concat = tf.reshape(st_concat, (n_states, units))
            st_ta = st_ta.write(step, st_concat)

            return step + 1, preds_ta, st_ta

        loop_vars = (0, predictions_ta, states_ta)

        final_step, final_predictions_ta, final_states_ta = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=loop_vars,
            parallel_iterations=1,
        )


        # Convert from TensorArray to Tensor
        predictions_out = tf.stop_gradient(
            final_predictions_ta.stack()
        )  # shape (steps+1, 1, 1, input_dim)
        predictions_out = tf.squeeze(predictions_out, axis=1)  # (steps+1, 1, input_dim)
        predictions_out = tf.transpose(
            predictions_out, [1, 0, 2]
        )  # (1, steps+1, input_dim)

        # Convert TensorArray to Tensor for states if not None
        states_out = tf.stop_gradient(
            final_states_ta.stack()
        )  # shape (steps, n_states, units)
        states_out = tf.transpose(
            states_out, [1, 0, 2]
        )  # shape (n_states, steps, units)

        return predictions_out, states_out

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input: tf.Tensor) -> tf.Tensor:
        """Forecasts a single step.

        Args:
            current_input (tf.Tensor): Current input tensor.

        Returns:
            tf.Tensor: Forecasted output tensor.
        """
        return self.call(current_input)

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
        reservoir = keras.layers.deserialize(config.pop("reservoir"))
        readout = keras.layers.deserialize(config.pop("readout"))
        seed = config.pop("seed", None)
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
        input_shape = self.get_build_config()["input_shape"]
        input_shape = tuple(input_shape)
        inputs = keras.Input(batch_shape=input_shape, name="Input_Layer")
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return keras.utils.plot_model(model, **kwargs)

    def get_build_config(self) -> dict:
        """
        Returns a dictionary containing everything needed to rebuild this model
        when deserialized. In your code, 'plot()' relies on 'get_build_config()["input_shape"]',
        so we must include it here.
        """
        # If the model hasn't been built yet, self._input_shape may be None.
        # In that case, return an empty dict or handle as needed.
        return {
            "input_shape": self._input_shape
        } if getattr(self, "_input_shape", None) is not None else {}

    def build_from_config(self, config):
        """Build the model from the given configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        self.built = True # TODO: Perhaps this is a patch and a hideous solution

@keras.saving.register_keras_serializable(
    package="ReservoirComputers", name="ReservoirEnsemble"
)
class ReservoirEnsemble(keras.Model):
    """
    Reservoir Ensemble model.

    Args:
        reservoir_computers (List[ReservoirComputer]): List of Reservoir Computers in the ensemble.
        seed (int | None): The seed of the model.
    """

    def __init__(self, reservoir_computers: List[ReservoirComputer], **kwargs):
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
        reservoir_computer_output_shape = self.reservoir_computers[
            0
        ].compute_output_shape(input_shape)
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
        reservoir_outputs = [
            reservoir_computer(inputs)
            for reservoir_computer in self.reservoir_computers
        ]
        output = self.outlier_removal_layer(reservoir_outputs)

        return output

    def train(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        train_target: np.ndarray,
        regularization: float,
        log: bool = False,
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

    @tf.function
    def ensure_ESP(self, transient_data: tf.Tensor) -> None:
        """
        Ensures ESP on each ReservoirComputer by simply calling their ensure_ESP methods.
        """
        for rc in self.reservoir_computers:
            rc.ensure_ESP(transient_data)

    def forecast(
        self,
        forecast_length: int,
        forecast_transient_data: np.ndarray,
        val_data: np.ndarray,
        store_states: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], np.ndarray, Optional[int]]:
        """
        Forecast the ensemble for a given number of steps using tf.while_loop, mirroring
        the same performance approach used in ReservoirComputer.

        This method resets states, ensures ESP, then calls _perform_forecasting_fast_with_states().
        The 'error_threshold' logic is omitted here for clarity; adapt as needed.

        Args:
            forecast_length: How many timesteps to forecast.
            forecast_transient_data: Data used to washout the reservoir states (ensure ESP).
            val_data: The actual data from which to start forecasting.
            val_target: The true target data (unused here, but left for API compatibility).
            store_states: Whether to store internal states at each forecast step.
            error_threshold: Currently unused, but kept for API compatibility.

        Returns:
            (predictions_out, states_out, empty_error_array, None)
            - predictions_out: shape (1, forecast_length, output_dim)
            - states_out: shape (total_states, forecast_length, units) if store_states=True, else None
            - empty_error_array: placeholder (np.array([])) for error measures
            - None: placeholder for steps_to_exceed_threshold
        """
        self.reset_states()

        if forecast_length > val_data.shape[1]:
            logging.info("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        # Prepare Tensors
        initial_point = val_data[:, :1, :]  # shape (1,1,input_dim)
        forecast_transient_data_tf = tf.convert_to_tensor(
            forecast_transient_data, dtype=tf.float32
        )
        initial_point_tf = tf.convert_to_tensor(initial_point, dtype=tf.float32)

        # Run the fast forecasting loop
        predictions_out, states_out = self._perform_forecasting_fast_with_states(
            initial_point_tf,
            forecast_transient_data_tf,
            steps=forecast_length,
            store_states=store_states,
        )

        # Remove the initial step from the predictions so that shape = (1, forecast_length, output_dim)
        predictions_out = predictions_out[:, 1:, :]

        if store_states:
            return predictions_out, states_out
        return predictions_out, None

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input: tf.Tensor) -> tf.Tensor:
        """Forecasts a single step.

        Args:
            current_input (tf.Tensor): Current input tensor.

        Returns:
            tf.Tensor: Forecasted output tensor.
        """
        return self.call(current_input)

    @tf.function
    def _perform_forecasting_fast_with_states(
        self,
        initial_point: tf.Tensor,
        forecast_transient_data: tf.Tensor,
        steps: int,
        store_states: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Analogous to ReservoirComputer._perform_forecasting_fast_with_states(), but for multiple
        ReservoirComputers. Uses tf.while_loop to avoid Python overhead.

        Args:
            initial_point:  (1, 1, input_dim)
            forecast_transient_data:  (transient_length, 1, input_dim)
            steps: Number of forecast steps.
            store_states: Whether to store internal states for each step.

        Returns:
            predictions_out: shape (1, steps+1, output_dim)
            states_out: shape (total_states, steps, units) if store_states=True, else None
        """
        # 1) Ensure ESP for all ReservoirComputers
        self.ensure_ESP(forecast_transient_data)

        # 2) Prepare TensorArrays for predictions and states
        # We'll write shape (1,1,output_dim) at each step, plus the initial.
        predictions_ta = tf.TensorArray(
            dtype=tf.float32,
            size=steps + 1,
            element_shape=(1, 1, initial_point.shape[2]),
        )
        # Put the initial_point in index 0
        predictions_ta = predictions_ta.write(0, initial_point)

        states_ta = None
        if store_states:
            # We'll gather states from each ReservoirComputer’s get_states()
            # Each state is typically shape (1, units).
            # We'll flatten them all into a single [sum_of_states, units].
            # That shape is stored at each step.
            st_list = []
            for rc in self.reservoir_computers:
                st_list.extend(rc.get_states())
            total_states = len(st_list)
            units = st_list[0].shape[1]  # assume all have shape (1, units)

            states_ta = tf.TensorArray(
                dtype=tf.float32,
                size=steps,
                element_shape=(total_states, units),
            )

        # 3) Define the loop condition and body
        def loop_cond(step, preds_ta, st_ta):
            return step < steps

        def loop_body(step, preds_ta, st_ta):
            # Read the last prediction
            current_input = preds_ta.read(step)  # shape (1,1,input_dim)
            # Forecast next step
            new_prediction = self.forecast_step(current_input)  # shape (1,1,output_dim)
            preds_ta = preds_ta.write(step + 1, new_prediction)

            if store_states:
                # Collect the states from every ReservoirComputer
                st_list = []
                for rc in self.reservoir_computers:
                    st_list.extend(rc.get_states())
                # Concatenate each state (shape (1, units)) along axis=0 => (total_states, units)
                st_concat = tf.concat([s[0] for s in st_list], axis=0)
                st_ta = st_ta.write(step, st_concat)

            return step + 1, preds_ta, st_ta

        # 4) Run tf.while_loop
        step_init = tf.constant(0)
        loop_vars = (step_init, predictions_ta, states_ta)
        final_step, final_predictions_ta, final_states_ta = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=loop_vars,
            parallel_iterations=1,
        )

        # 5) Convert predictions from TensorArray to Tensor
        predictions_out = tf.stop_gradient(
            final_predictions_ta.stack()
        )  # (steps+1, 1,1, output_dim)
        predictions_out = tf.squeeze(
            predictions_out, axis=1
        )  # (steps+1, 1, output_dim)
        predictions_out = tf.transpose(
            predictions_out, [1, 0, 2]
        )  # (1, steps+1, output_dim)

        # 6) Convert states if requested
        states_out = tf.stop_gradient(
            final_states_ta.stack()
        )  # shape (steps, total_states, units)
        states_out = tf.transpose(states_out, [1, 0, 2])  # (total_states, steps, units)

        return predictions_out, states_out

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
        return [
            reservoir_computer.get_states()
            for reservoir_computer in self.reservoir_computers
        ]

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
                "reservoir_computers": [
                    keras.utils.serialize_keras_object(rc)
                    for rc in self.reservoir_computers
                ],
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
        reservoir_computers = [
            keras.layers.deserialize(reservoir_computer)
            for reservoir_computer in config.pop("reservoir_computers")
        ]
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
        input_shape = self.get_build_config()["input_shape"]
        input_shape = tuple(input_shape)
        inputs = keras.Input(batch_shape=input_shape, name="Input_Layer")
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return keras.utils.plot_model(model, **kwargs)

    def get_build_config(self) -> dict:
        """
        Returns a dictionary containing everything needed to rebuild this model
        when deserialized. In your code, 'plot()' relies on 'get_build_config()["input_shape"]',
        so we must include it here.
        """
        # If the model hasn't been built yet, self._input_shape may be None.
        # In that case, return an empty dict or handle as needed.
        return {
            "input_shape": self._input_shape
        } if getattr(self, "_input_shape", None) is not None else {}

    def build_from_config(self, config):
        """Build the model from the given configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        self.built = True
