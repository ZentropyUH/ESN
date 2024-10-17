import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from rich.progress import track
from typing import Any
from typing import Tuple
from typing import Optional

import keras
import tensorflow as tf
from keras import Model


from keras import Layer

from src.customs.custom_reservoirs import BaseReservoir
from src.utils import timer
from src.utils import TF_Ridge
class ReservoirComputer:
    """
    Base Reservoir Computer model.\n
    Works as a base class for different types of Reservoir Computers. The key components will be a reservoir and a readout layer.
    Groups a set of basic functionalities for all Reservoir Computers.

    Args:
        reservoir (keras.Layer): The reservoir layer of the Reservoir Computer.

        readout (keras.Layer): The readout layer of the Reservoir Computer.
    """
    def __init__(
        self,
        reservoir: BaseReservoir,
        readout: Layer,
        seed: int | None
    ) -> None:
        """
        Initialize the Reservoir Computer model.

        Args:
            reservoir (BaseReservoir): The reservoir layer of the model.
            readout (Layer): The readout layer of the model.
            seed (int): The seed of the
        """
        self.reservoir = reservoir
        self.readout = readout
        # Raise a warning if the seed is None
        if seed is None:
            print("Seed is None. Reproducibility is not guaranteed.")
        self.seed = seed

        # This is a flag to check if the reservoir and readout are built (loading a previously trained model)
        self._built = self.readout.built and self.reservoir.built

        self.model = self._create_model(
            reservoir=self.reservoir,
            readout=self.readout
            )

    def _create_model(self, reservoir, readout) -> Model:
        """
        Create the model with the reservoir and readout layers.

        Args:
            reservoir (BaseReservoir): The reservoir layer of the model.
            readout (Layer): The readout layer of the model.

        Returns:
            Model: The model with the reservoir and readout layers.
        """
        if reservoir.built and readout.built:
            model = keras.Sequential([reservoir, readout], name="RC_model")
            return model

    @classmethod
    def from_model(cls, model: Model, seed: int) -> "ReservoirComputer":
        """
        Load a model from a Keras model.

        Args:
            model (Model): The model to load.
            seed (int): The seed of the model.

        Returns:
            ReservoirComputer: The loaded Reservoir Computer model.
        """
        if len(model.layers) != 2:
            raise RuntimeError(f"The loaded model has {len(model.layers)} layers. Should have only two, reservoir and readout.")

        reservoir = model.layers[0]
        readout = model.layers[1]

        return cls(reservoir=reservoir, readout=readout, seed=seed)

    # Check if this is correct
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model must be trained to predict")
        return self.model(*args, **kwargs)

    def get_weights(self) -> list:
        """
        Get the weights of the model.

        Returns:
            list: List of weights of the model.
        """
        return self.model.get_weights()

    def predict(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """
        Predicts the output of the model for the given inputs.

        Args:
            inputs (tf.Tensor): The inputs to predict.

        Returns:
            tf.Tensor: The predicted output of the model.

        Raises:
            Exception: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained to predict")
        return self.model.predict(inputs, **kwargs)

    def train(
        self,
        transient_data: np.ndarray,
        train_data: np.ndarray,
        train_target: np.ndarray,
        regularization: float,
    ) -> None:
        """
        Training process of the model.

        Args:
            transient_data (tf.Tensor): Transient data.
            train_data (tf.Tensor): Data to train the model.
            train_target (tf.Tensor): Target data for the training.
            regularization (float): Regularization value for linear readout.

        Return:
            None
        """
        harvested_states = self._harvest(transient_data, train_data)

        training_loss = self._calculate_readout(
            harvested_states, train_target, regularization
        )

        self.model = self._create_model(
            reservoir=self.reservoir,
            readout=self.readout
            )

        self._built = True

        return training_loss

    def _harvest(
        self, transient_data: tf.Tensor, train_data: tf.Tensor
    ) -> tf.Tensor:
        """
        This method ensures ESP by predicting the transient data and then harvests the reservoir states. It returns the harvested reservoir states.

        Args:
            transient (tf.Tensor): Transient data to ensure ESP.

            data (tf.Tensor): Data to harvest the reservoir states.

        Returns:
            tf.Tensor: The harvested reservoir states.
        """
        if not self.reservoir.built:
            self.reservoir.build(input_shape=transient_data.shape)

        # Combine transient and train data
        combined_data = tf.concat([transient_data, train_data], axis=1)

        with timer("Harvesting states with combined data"):
            harvested_states = self.reservoir.predict(combined_data, verbose=0)

        # Extract the states corresponding to the training data (Discard the transient data predictions)
        harvested_states = harvested_states[:, transient_data.shape[1]:, :]

        return harvested_states

    def _calculate_readout(
        self,
        harvested_states: tf.Tensor,
        train_target: tf.Tensor,
        regularization: float = 0,
    ) -> float:
        """
        Calculate the readout of the model using Ridge regression with SVD solver.

        Args:
            harvested_states (tf.Tensor): The harvested reservoir states.

            train_target (tf.Tensor): The target data for the training.

            regularization (float): Regularization value for linear readout.

        Returns:
            float: The training loss of the model.
        """

        print("Harvested states shape: ", harvested_states.shape)
        print("Train target shape: ", train_target.shape)

        harvested_states = tf.cast(harvested_states, dtype=tf.float64)
        train_target = tf.cast(train_target, dtype=tf.float64)

        regressor = TF_Ridge(alpha=regularization)

        with timer("Fitting the regressor"):
            regressor.fit(harvested_states, train_target)

        coef = regressor._coef
        intercept = regressor._intercept

        # Set weights of the readout layer
        if not self.readout.built:
            self.readout.build(harvested_states.shape)
        self.readout.set_weights([coef, intercept])

        # Calculate training loss
        predictions = regressor.predict(harvested_states)
        training_loss = tf.reduce_mean(tf.square(predictions - train_target))

        return training_loss

    # TODO: Reimplement the train_several method
    def train_several(
        self,
        transient_data_array: tf.Tensor,
        train_data_array: tf.Tensor,
        train_target_array: tf.Tensor,
        regularization: float,
    ) -> None:
        """
        Training process of the model.

        Args:
            transient_data_array (tf.Tensor): Transient data. The shape is [datasets, 1, timesteps, features]
            train_data_array (tf.Tensor): Data to train the model. The shape is [datasets, 1, timesteps, features]
            train_target_array (tf.Tensor): Target data for the training. The shape is [datasets, 1, timesteps, features]
            regularization (float): Regularization value for linear readout.

        Return:
            None
        """
        harvested_states = []
        for transient_data, train_data in zip(transient_data_array, train_data_array):
            _harvested_states = self._harvest(transient_data, train_data)
            harvested_states.append(_harvested_states)

        harvested_states = tf.concat(harvested_states, axis=1)

        train_target = tf.concat(train_target_array, axis=1)

        training_loss = self._calculate_readout(
            harvested_states, train_target, regularization
        )

        self.model = self._create_model(
            reservoir=self.reservoir,
            readout=self.readout
            )

        self._built = True

        return training_loss

    def get_states(self) -> list:
        """Retrieve the current states of all RNN layers in the model.

        Returns:
            list: List of states.
        """
        return self.reservoir.get_states()

    # TODO: Check if this is correct
    def set_states(self, new_states: list) -> None:
        """Set the states of all RNN layers in the model.

        Args:
            new_states (list): List of states to set.

        Returns:
            None
        """
        self.reservoir.set_states(new_states)

    def set_random_states(self, threshold: float = 1, seed: Optional[int] = None) -> None:
        """Set random states to the reservoir. The states are generated with values in [-threshold, threshold].

        Args:
            threshold (float, optional): The threshold for the random states. Defaults to 1.
            seed (Optional[int], optional): The seed for the random generator. Defaults to None.

        Raises:
            RuntimeError: If the reservoir is not built.
            ValueError: If no states are found to reset.

        Returns:
            None
        """
        # Ensure the reservoir is built
        if not self.reservoir.built:
            raise RuntimeError("The reservoir must be built before setting states.")

        current_states = self.get_states()

        if not current_states:
            raise ValueError("No states found to reset. Ensure the reservoir is stateful and built.")

        # Create random generator with or without seed
        if seed is not None:
            tf_rng = tf.random.Generator.from_seed(seed)
        else:
            tf_rng = tf.random.Generator.from_non_deterministic_state()

        new_random_states = []

        for state in current_states:
            # Generate a random state with the same shape as the current state, with values in [-threshold, threshold]
            random_state = tf_rng.uniform(
                state.shape, minval=-threshold, maxval=threshold, dtype=state.dtype
            )
            # Append the random state tensor directly (no need for Variable)
            new_random_states.append(random_state)

        # Set the new random states manually
        self.reservoir.set_states(new_random_states)

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
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Forecast the model for a given number of steps. It also calculates the cumulative error if the error threshold is specified. If internal_states is True, it will store the internal states of the RC over time. The method returns the predictions, the states over time, the cumulative RMSE, and the number of steps to exceed the error threshold.

        Args:
            forecast_length (int): The number of steps to forecast.
            forecast_transient_data (np.ndarray): The transient data to ensure ESP.
            val_data (np.ndarray): The validation data to forecast.
            val_target (np.ndarray): The
            internal_states (bool, optional): Whether to store the internal states of the RC over time. Defaults to False.
            error_threshold (Optional[float], optional): The threshold for the cumulative RMSE to record the number of steps to exceed it. Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]: A tuple containing the predictions array, the states over time array, the cumulative error array, and the number of steps to exceed the error threshold.
        """
        self.reset_states()

        if forecast_length > val_data.shape[1]:
            print("Truncating the forecast length to match the data.")
        forecast_length = min(forecast_length, val_data.shape[1])

        predictions, states_over_time = self._initialize_forecast_structures(
            val_data, forecast_length, internal_states
        )

        cumulative_error, steps_to_exceed_threshold, states_over_time = (
            self._perform_forecasting(
                predictions,
                forecast_transient_data,
                val_target,
                error_threshold,
                states_over_time,
            )
        )

        predictions = predictions[:, 1:, :]

        return (
            predictions,
            states_over_time,
            cumulative_error,
            steps_to_exceed_threshold,
        )

    def _initialize_forecast_structures(
        self, val_data: np.ndarray, forecast_length: int, internal_states: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """This method initializes the structures to store the predictions and the internal states of the RC over time. It also initializes the first prediction with the first data point of the validation data.

        Args:
            val_data (np.ndarray): The validation data to forecast.
            forecast_length (int): The number of steps to forecast.
            internal_states (bool): Whether to store the internal states of the RC over time.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: A tuple containing the predictions array and the states over time array. The predictions array has shape (1, forecast_length + 1, val_data.shape[2]). The states over time array has shape (forecast_length, self.model.get_layer("esn_rnn").cell.units) if internal_states is True, otherwise it is None.
        """
        predictions = tf.Variable(
            tf.zeros((1, forecast_length + 1, val_data.shape[2])), dtype=tf.float32
        )
        predictions[:, :1, :].assign(
            tf.convert_to_tensor(val_data[:, :1, :], dtype=tf.float32)
        )  # Initialize the first prediction with the first data point

        states_over_time = (
            tf.TensorArray(tf.float32, size=forecast_length, dynamic_size=True)
            if internal_states
            else None
        )

        return predictions, states_over_time

    # @tf.function(reduce_retracing=True)
    def _perform_forecasting(
        self,
        predictions: np.ndarray,
        forecast_transient_data: np.ndarray,
        val_target: np.ndarray,
        error_threshold: Optional[float],
        states_over_time: Optional[np.ndarray] = None,
    ):
        """Perform the forecasting process for the model. This method does the forecasting and calculates the cumulative error if the error threshold is specified. If states_over_time is not None, it will store the internal states of the RC over time.

        Args:
            predictions (np.ndarray): The predictions array to store the forecasted data. The first element of the array must be the first data point of the validation data. Its shape will be (1, forecast_length + 1, n_features).
            forecast_transient_data (np.ndarray): This is the transient data to ensure ESP. It is used to make the internal states of the RC converge to the true states of the data.
            val_target (np.ndarray): The target data of the validation data. It is used to calculate the cumulative error.
            error_threshold (Optional[float]): The threshold for the cumulative RMSE to stop the forecasting process.
            states_over_time (Optional[np.ndarray], optional): The array to store the internal states of the RC over time. Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[int], Optional[np.ndarray]]: A tuple containing the cumulative error, the number of steps to exceed the error threshold, and the states over time array.
        """
        # self.predict(forecast_transient_data)  # Ensure ESP by predicting transient data
        self.predict(forecast_transient_data, verbose=0)

        squared_errors_sum = tf.Variable(0.0, dtype=tf.float32)

        cumulative_error = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        steps_to_exceed_threshold = 0 if error_threshold is not None else None

        for step in track(
            range(predictions.shape[1] - 1), description="Forecasting..."
        ):

            current_input = predictions[:, step : step + 1, :]
            new_prediction = self.forecast_step(current_input)
            predictions[:, step + 1 : step + 2, :].assign(new_prediction)

            if states_over_time is not None:
                state = self.model.get_layer("esn_rnn").states[0]
                states_over_time = states_over_time.write(step, state)

            if error_threshold is not None:  # Calculate the RMSE at the current step

                # Update squared errors sum
                current_squared_errors = tf.reduce_sum(
                    tf.square(val_target[:, step, :] - predictions[:, step + 1, :])
                )
                squared_errors_sum.assign_add(current_squared_errors)

                # Calculate cumulative RMSE from the start to the current step
                current_rmse = tf.sqrt(
                    squared_errors_sum / tf.cast(step + 1, tf.float32)
                )
                cumulative_error = cumulative_error.write(step, current_rmse)

                if current_rmse > error_threshold and steps_to_exceed_threshold == 0:
                    steps_to_exceed_threshold = step - 1

        return cumulative_error, steps_to_exceed_threshold, states_over_time

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input):
        """
        Forecast a single step using the model.
        Wrapped with tf.function for performance optimization.

        Args:
            current_input (tf.Tensor): The current input to forecast.

        Returns:
            tf.Tensor: The forecasted output for the current input.
        """
        return self.model(current_input)

    def save(self, path: str) -> None:
        """
        Save the model in a folder.

        Args:
            path (str): The destination folder to save all the files of the model.

        Returns:
            None
        """
        # Create the dir if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, "model.keras"), include_optimizer=False)

        with open(os.path.join(path, "seed.txt"), "w") as f:
            f.write(str(self.seed))

    # TODO: Implement the plot_model method
    def plot_model(self, **kwargs):
        keras.utils.plot_model(self.model, **kwargs)

    @staticmethod
    def load(path: str) -> "ReservoirComputer":
        """
        Load the model from folder format.

        Args:
            path (str): Folder to load the model.

        Return:
            model (ReservoirComputer): Return the loaded instance of the RC model.
        """
        model_path = os.path.join(path, "model.keras")

        model: keras.Model = keras.models.load_model(model_path, compile=False)

        seed_path = os.path.join(path, "seed.txt")
        with open(seed_path, "r") as f:
            seed = int(f.read())

        esn = ReservoirComputer.from_model(model=model, seed=seed)

        return esn


# region Reservoirs TODO: Implement the parallel ESN and ECA ESN using the new structure

# def generate_Parallel_ESN(
#     units: int,
#     partitions: int = 1,
#     overlap: int = 0,
#     leak_rate: float = 1.0,
#     features: int = 1,
#     activation: str = "tanh",
#     input_reservoir_init: str = "InputMatrix",
#     input_bias_init: str = "random_uniform",
#     reservoir_kernel_init: str = "WattsStrogatzNX",
#     exponent: int = 2,
#     seed: int = None,
# ) -> ESN:
#     """
#     Assemble all the layers in a parallel ESN model.

#     Args:
#         units (int): Number of units in the reservoir.

#         partitions (int): Number of partitions of the reservoir.

#         overlap (int): Number of overlapping units between partitions.

#         leak_rate (float): Leak rate of the reservoir.

#         features (int): Number of features of the input data.

#         activation (str): Activation function of the reservoir.

#         input_reservoir_init (str): Initialization method for the input matrix.

#         input_bias_init (str): Initialization method for the input bias.

#         reservoir_kernel_init (str): Initialization method for the reservoir matrix.

#         exponent (int): Exponent of the input matrix.

#         seed (int): Seed for the random number generator.

#     Return:
#         model (ESN): Return the loaded instance of the ESN model.
#     """
#     if seed is None:
#         seed = np.random.randint(0, 1000000)
#     print(f"\nSeed: {seed}\n")
#     np.random.seed(seed)
#     tf.random.set_seed(seed)

#     reservoir = parallel_esn(
#         units=units,
#         partitions=partitions,
#         overlap=overlap,
#         leak_rate=leak_rate,
#         activation=activation,
#         features=features,
#         input_reservoir_init=input_reservoir_init,
#         input_bias_init=input_bias_init,
#         reservoir_kernel_init=reservoir_kernel_init,
#         exponent=exponent,
#     )

#     readout_layer = keras.layers.Dense(
#         features, activation="linear", name="readout", trainable=False
#     )

#     model = ESN(reservoir=reservoir, readout=readout_layer, seed=seed)
#     return model


# def generate_ECA_ESN(
#     units: int,
#     rule: Union[str, int, np.ndarray, list, tf.Tensor] = 110,
#     steps: int = 1,
#     leak_rate: float = 1.0,
#     features: int = 1,
#     activation: str = "tanh",
#     input_reservoir_init: str = "InputMatrix",
#     input_bias_init: str = "random_uniform",
#     exponent: int = 2,
#     seed: int = None,
# ) -> ESN:
#     """
#     Assemble all the layers in an ECA ESN model.

#     Args:
#         units (int): Number of units in the reservoir.

#         rule (Union[str, int, np.ndarray, list, tf.Tensor]): The rule to use for the ECA reservoir.

#         steps (int): Number of steps to run the ECA reservoir.

#         leak_rate (float): Leak rate of the reservoir.

#         features (int): Number of features of the input data.

#         activation (str): Activation function of the reservoir.

#         input_reservoir_init (str): Initialization method for the input matrix.

#         input_bias_init (str): Initialization method for the input bias.

#         exponent (int): Exponent of the input matrix.

#         seed (int): Seed for the random number generator.

#     Return:
#         model (ESN): Return the loaded instance of the ESN model.
#     """
#     if seed is None:
#         seed = np.random.randint(0, 1000000)
#     print(f"\nSeed: {seed}\n")
#     np.random.seed(seed)
#     tf.random.set_seed(seed)

#     reservoir = eca_esn(
#         units=units,
#         rule=rule,
#         steps=steps,
#         leak_rate=leak_rate,
#         activation=activation,
#         features=features,
#         input_reservoir_init=input_reservoir_init,
#         input_bias_init=input_bias_init,
#         exponent=exponent,
#     )

#     readout_layer = keras.layers.Dense(
#         features, activation="linear", name="readout", trainable=False
#     )

#     model = ESN(reservoir, readout_layer)
#     return model


# region Reimplement above


# def parallel_esn(
#     units: int,
#     leak_rate: float = 1,
#     features: int = 1,
#     activation: str = "tanh",
#     input_reservoir_init: str = "InputMatrix",
#     input_bias_init: str = "random_uniform",
#     reservoir_kernel_init: str = "WattsStrogatzNX",
#     exponent: int = 2,
#     partitions: int = 1,
#     overlap: int = 0,
# ):

#     # FIX
#     assert features % partitions == 0, "Input length must be divisible by partitions"

#     assert (
#         features // partitions > overlap
#     ), "Overlap must be smaller than the length of the partitions"

#     inputs = keras.Input(batch_shape=(1, None, features), name="Input")

#     inputs_splitted = InputSplitter(
#         partitions=partitions, overlap=overlap, name="splitter"
#     )(inputs)

#     # Create the reservoirs
#     reservoir_outputs = []
#     for i in range(partitions):

#         esn_cell = EsnCell(
#             units=units,
#             name="EsnCell",
#             activation=activation,
#             leak_rate=leak_rate,
#             input_initializer=input_reservoir_init,
#             input_bias_initializer=input_bias_init,
#             reservoir_initializer=reservoir_kernel_init,
#         )

#         reservoir = keras.layers.RNN(
#             esn_cell,
#             trainable=False,
#             stateful=True,
#             return_sequences=True,
#             name=f"esn_rnn_{i}",
#         )

#         reservoir_output = reservoir(inputs_splitted[i])
#         reservoir_output = PowerIndex(exponent=exponent, index=i, name=f"pwr_{i}")(
#             reservoir_output
#         )
#         reservoir_outputs.append(reservoir_output)

#     # Concatenate the power indices
#     output = keras.layers.Concatenate(name="esn_rnn")(reservoir_outputs)

#     output = keras.layers.Concatenate(name="Concat_ESN_input")([inputs, output])

#     parallel_reservoir = keras.Model(
#         inputs=inputs,
#         outputs=output,
#     )

#     return parallel_reservoir


# def eca_esn(
#     units: int,
#     leak_rate: float = 1,
#     features: int = 1,
#     activation: str = "tanh",
#     input_reservoir_init: str = "InputMatrix",
#     input_bias_init: str = "random_uniform",
#     rule: int = 110,
#     steps: int = 1,
#     exponent: int = 2,
# ):

#     eca_function = create_automaton_tf(rule, steps=steps)

#     inputs = keras.Input(batch_shape=(1, None, features), name="Input")

#     eca_cell = ReservoirCell(
#         units=units,
#         reservoir_function=eca_function,
#         input_initializer=input_reservoir_init,
#         input_bias_initializer=input_bias_init,
#         activation=activation,
#         leak_rate=leak_rate,
#         name="EcaCell",
#     )

#     eca_rnn = keras.layers.RNN(
#         eca_cell,
#         trainable=False,
#         stateful=True,
#         return_sequences=True,
#         name="esn_rnn",
#     )(inputs)

#     power_index = PowerIndex(exponent=exponent, index=2, name="pwr")(eca_rnn)

#     output = keras.layers.Concatenate(name="Concat_ESN_input")([inputs, power_index])

#     reservoir = keras.Model(
#         inputs=inputs,
#         outputs=output,
#     )

#     return reservoir
# endregion
