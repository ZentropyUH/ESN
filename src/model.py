import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from time import time
from rich.progress import track
from typing import Any
from typing import Tuple
from typing import Optional
from typing import Union
from typing import Callable
from typeguard import typechecked

import keras
import tensorflow as tf
from keras import Model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from keras import Initializer
from keras import Layer

from src.customs.custom_layers import EsnCell, PowerIndex
from src.utils import calculate_nrmse
from src.utils import calculate_rmse
from src.utils import TF_RidgeRegression
from src.utils import timer


# TODO: Add log
# TODO: separate prediction and evaluation
class ESN:
    """
    Base ESN model.\n
    Works as an assembly class for the layers that are passed to it as input.
    Groups a set of basic functionalities for an ESN.

    Args:
        reservoir (keras.Layer): The reservoir layer of the ESN.

        readout (keras.Layer): The readout layer of the ESN.
    """

    def __eq__(self, other):
        if not isinstance(other, ESN) or self.model is None or other.model is None:
            return False
        if self is other:
            return True  # Fast path for comparison with itself
        return all(
            np.array_equal(w1, w2)
            for w1, w2 in zip(self.model.get_weights(), other.model.get_weights())
        )

    def __hash__(self):
        """
        Generate a hash based on the weights of the internal Keras model.

        This method flattens and converts all weights to a bytes object,
        which is then used to compute a hash.

        Returns:
            int: The hash value of the model's weights.
        """
        if not hasattr(self, "_hash") or self._hash is None:
            if self.model is not None:
                # Compute hash only once after training
                weight_bytes = b"".join(w.tobytes() for w in self.model.get_weights())
                self._hash = hash(weight_bytes)
            else:
                self._hash = hash(("ESN", None))
        return self._hash

    @typechecked
    def __init__(self, reservoir: Layer, readout: Layer, seed: int | None) -> None:

        self.readout: Layer = readout
        self.reservoir: Layer = reservoir
        # Raise a warning if the seed is None
        if seed is None:
            print("Seed is None. Reproducibility is not guaranteed.")
        self.seed = seed

        # This is a flag to check if the reservoir and readout are built (loading a previously trained model)
        self._built = self.readout.built and self.reservoir.built

        if self._built:
            self.model = keras.Model(
                inputs=self.reservoir.inputs,
                outputs=self.readout(self.reservoir.output),
                name="ESN",
            )
        else:
            self.model: Model = None

    @classmethod
    def from_model(cls, model: Model, seed: int) -> None:

        resrvoir_inputs = model.get_layer("Input").output
        reservoir_outputs = model.get_layer("Concat_ESN_input").output
        readout = model.get_layer("readout")
        reservoir = keras.Model(
            inputs=resrvoir_inputs,
            outputs=reservoir_outputs,
        )

        return cls(reservoir=reservoir, readout=readout, seed=seed)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model must be trained to predict")
        return self.model(*args, **kwds)

    def get_weights(self):
        return self.model.get_weights()

    def predict(self, inputs: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predicts the output of the model for the given inputs.

        Args:
            inputs (np.ndarray): The input data to predict the output for.

        Returns:
            np.ndarray: The predicted output for the given inputs.

        Raises:
            Exception: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained to predict")
        return self.model.predict(inputs, **kwargs)

    def _harvest(
        self, transient_data: np.ndarray, train_data: np.ndarray
    ) -> np.ndarray:
        """
        This method ensures ESP by predicting the transient data and then harvests the reservoir states. It returns the harvested reservoir states.

        Args:
            transient (np.ndarray): Transient data to ensure ESP.

            data (np.ndarray): Data to harvest the reservoir states.

        Returns:
            np.ndarray: The harvested reservoir states.
        """
        if not self.reservoir.built:
            self.reservoir.build(input_shape=transient_data.shape)

        # print("\nEnsuring ESP...\n")
        with self.timer("Ensuring ESP"):
            self.reservoir.predict(transient_data)

        with self.timer("Harvesting"):
            harvested_states = self.reservoir.predict(train_data)

        return harvested_states

    def _calculate_readout(
        self,
        harvested_states: np.ndarray,
        train_target: np.ndarray,
        regularization: float = 0,
    ) -> float:
        """
        Calculate the readout of the model. This method uses Ridge regression to calculate the readout and set the weights of the readout layer. Will return the training loss of the model (RMSE).

        Args:
            harvested_states (np.ndarray): The harvested reservoir states.

            train_target (np.ndarray): The target data for the training.

            regularization (float): Regularization value for linear readout.

        Returns:
            float: The training loss of the model.
        """
        with self.timer("Calculating readout"):
            readout = Ridge(alpha=regularization, tol=0, solver="svd")
            readout.fit(harvested_states[0], train_target[0])

        predictions = readout.predict(harvested_states[0])

        training_loss = np.mean((predictions - train_target[0]) ** 2)

        print(f"Training loss: {training_loss}\n")

        # this is temporary
        nrmse = calculate_nrmse(target=train_target[0], prediction=predictions)
        print(f"NRMSE: {nrmse}\n")

        if not self.readout.built:
            self.readout.build(harvested_states[0].shape)

        self.readout.set_weights([readout.coef_.T, readout.intercept_])

        return training_loss

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
            transient_data (np.ndarray): Transient data.
            train_data (np.ndarray): Data to train the model.
            train_target (np.ndarray): Target data for the training.
            regularization (float): Regularization value for linear readout.

        Return:
            None
        """
        harvested_states = self._harvest(transient_data, train_data)

        training_loss = self._calculate_readout(
            harvested_states, train_target, regularization
        )

        self.model = keras.Model(
            inputs=self.reservoir.inputs,
            outputs=self.readout(self.reservoir.output),
            name="ESN",
        )

        self._built = True

        return training_loss

    def train_several(
        self,
        transient_data_array: np.ndarray,
        train_data_array: np.ndarray,
        train_target_array: np.ndarray,
        regularization: float,
    ) -> None:
        """
        Training process of the model.

        Args:
            transient_data_array (np.ndarray): Transient data.
            train_data_array (np.ndarray): Data to train the model.
            train_target_array (np.ndarray): Target data for the training.
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

        self.model = keras.Model(
            inputs=self.reservoir.inputs,
            outputs=self.readout(self.reservoir.output),
            name="ESN",
        )

        self._built = True

        return training_loss

    def get_states(self) -> list:
        """Retrieve the current states of all RNN layers in the model.

        Returns:
            list: List of states.
        """
        rnn_states = []
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.RNN):
                rnn_states.append(
                    layer.states[0]
                )  # For simple RNN, there's only one state tensor
        return rnn_states

    def set_states(self, states: list) -> None:
        """Set the states of all RNN layers in the model.

        Args:
            states (list): List of states to set.

        Returns:
            None
        """
        state_index = 0
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.RNN):
                if layer.states[0].shape == states[state_index].shape:
                    layer.states[0] = states[
                        state_index
                    ]  # For simple RNN, there's only one state tensor
                    state_index += 1
                else:
                    raise ValueError(
                        "The shape of the state tensor does not match the shape of the layer's state tensor."
                    )

    def set_random_states(
        self, threshold: float = 1, seed: Optional[int] = None
    ) -> None:
        """Set random states for all RNN layers in the model with values sampled uniformly from [-a, a]."""
        current_states = self.get_states()
        new_random_states = []

        if seed is not None:
            tf_rng = tf.random.Generator.from_seed(seed)
        else:
            tf_rng = tf.random.Generator.from_non_deterministic_state()

        for state in current_states:
            # Generate a random state with the same shape as the current state, with values in [-a, a]
            random_state = tf_rng.uniform(
                state.shape, minval=-threshold, maxval=threshold
            )
            new_random_states.append(random_state)

        self.set_states(new_random_states)

    def reset_states(self):
        """Reset internal states of the RNNs of the model."""
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.RNN):
                layer.reset_states()

            # function code here

    def forecast(
        self,
        forecast_length: int,
        forecast_transient_data: np.ndarray,
        val_data: np.ndarray,
        val_target: np.ndarray,
        internal_states: bool = False,
        error_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Forecast the model for a given number of steps. It also calculates the cumulative error if the error threshold is specified. If internal_states is True, it will store the internal states of the ESN over time. The method returns the predictions, the states over time, the cumulative RMSE, and the number of steps to exceed the error threshold.

        Args:
            forecast_length (int): The number of steps to forecast.
            forecast_transient_data (np.ndarray): The transient data to ensure ESP.
            val_data (np.ndarray): The validation data to forecast.
            val_target (np.ndarray): The
            internal_states (bool, optional): Whether to store the internal states of the ESN over time. Defaults to False.
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
        """This method initializes the structures to store the predictions and the internal states of the ESN over time. It also initializes the first prediction with the first data point of the validation data.

        Args:
            val_data (np.ndarray): The validation data to forecast.
            forecast_length (int): The number of steps to forecast.
            internal_states (bool): Whether to store the internal states of the ESN over time.

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
        """Perform the forecasting process for the model. This method does the forecasting and calculates the cumulative error if the error threshold is specified. If states_over_time is not None, it will store the internal states of the ESN over time.

        Args:
            predictions (np.ndarray): The predictions array to store the forecasted data. The first element of the array must be the first data point of the validation data. Its shape will be (1, forecast_length + 1, n_features).
            forecast_transient_data (np.ndarray): This is the transient data to ensure ESP. It is used to make the internal states of the ESN converge to the true states of the data.
            val_target (np.ndarray): The target data of the validation data. It is used to calculate the cumulative error.
            error_threshold (Optional[float]): The threshold for the cumulative RMSE to stop the forecasting process.
            states_over_time (Optional[np.ndarray], optional): The array to store the internal states of the ESN over time. Defaults to None.

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
                    steps_to_exceed_threshold = step + 1

        return cumulative_error, steps_to_exceed_threshold, states_over_time

    @tf.function(reduce_retracing=True)
    def forecast_step(self, current_input):
        """
        Forecast a single step using the model.
        Wrapped with tf.function for performance optimization.
        """
        return self.model(current_input)

    def save(self, path: str) -> None:
        """
        Save the model in a folder.

        Args:
            path (str): The destination folder to save all the files of the model.
        """
        # Create the dir if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, "model.keras"), include_optimizer=False)

        with open(os.path.join(path, "seed.txt"), "w") as f:
            f.write(str(self.seed))

    def plot_model(self, **kwargs):
        keras.utils.plot_model(self.model, **kwargs)

    @staticmethod
    def load(path: str) -> "ESN":
        """
        Load the model from folder format.

        Args:
            path (str): Folder to load the model.

        Return:
            model (ESN): Return the loaded instance of the ESN model.
        """
        model_path = os.path.join(path, "model.keras")

        model: keras.Model = keras.models.load_model(model_path, compile=False)

        seed_path = os.path.join(path, "seed.txt")
        with open(seed_path, "r") as f:
            seed = int(f.read())

        esn = ESN.from_model(model=model, seed=seed)

        return esn


# region Reservoirs


def simple_reservoir(
    units: int,
    leak_rate: float = 1,
    features: int = 1,
    activation: Union[str, Callable] = "tanh",
    input_reservoir_init: Union[str, Initializer] = "InputMatrix",
    input_bias_init: Union[str, Initializer] = "random_uniform",
    reservoir_kernel_init: Union[str, Initializer] = "WattsStrogatzNX",
    exponent: int = 2,
):

    inputs = keras.Input(batch_shape=(1, None, features), name="Input")

    esn_cell = EsnCell(
        units=units,
        name="EsnCell",
        activation=activation,
        leak_rate=leak_rate,
        input_initializer=input_reservoir_init,
        input_bias_initializer=input_bias_init,
        reservoir_initializer=reservoir_kernel_init,
    )

    esn_rnn = keras.layers.RNN(
        esn_cell,
        trainable=False,
        stateful=True,
        return_sequences=True,
        name="esn_rnn",
    )(inputs)

    power_index = PowerIndex(exponent=exponent, index=2, name="pwr")(esn_rnn)

    output = keras.layers.Concatenate(name="Concat_ESN_input")([inputs, power_index])

    reservoir = keras.Model(
        inputs=inputs,
        outputs=output,
    )

    return reservoir


# endregion


def generate_ESN(
    units: int,
    leak_rate: float = 1.0,
    features: int = 1,
    activation: Union[str, Callable] = "tanh",
    input_reservoir_init: Union[str, Initializer] = "InputMatrix",
    input_bias_init: Union[str, Initializer] = "random_uniform",
    reservoir_kernel_init: Union[str, Initializer] = "WattsStrogatzNX",
    exponent: int = 2,
    seed: int | None = None,
) -> ESN:
    """
    Assemble all the layers in ESN model.

    Args:
        units (int): Number of units in the reservoir.

        leak_rate (float): Leak rate of the reservoir.

        features (int): Number of features of the input data.

        activation (str): Activation function of the reservoir.

        input_reservoir_init (str|keras Initializer): Initializer for the input matrix.

        input_bias_init (str|keras Initializer): Initializer for the input bias.

        reservoir_kernel_init (str|keras Initializer): Initializer for the reservoir matrix.

        exponent (int): Exponent of the input matrix.

        seed (int): Seed for the random number generator.

    Return:
        model (ESN): Return the loaded instance of the ESN model.
    """
    reservoir = simple_reservoir(
        units=units,
        leak_rate=leak_rate,
        activation=activation,
        features=features,
        input_reservoir_init=input_reservoir_init,
        input_bias_init=input_bias_init,
        reservoir_kernel_init=reservoir_kernel_init,
        exponent=exponent,
    )

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = ESN(reservoir, readout_layer, seed=seed)
    return model


def generate_Parallel_ESN(
    units: int,
    partitions: int = 1,
    overlap: int = 0,
    leak_rate: float = 1.0,
    features: int = 1,
    activation: str = "tanh",
    input_reservoir_init: str = "InputMatrix",
    input_bias_init: str = "random_uniform",
    reservoir_kernel_init: str = "WattsStrogatzNX",
    exponent: int = 2,
    seed: int = None,
) -> ESN:
    """
    Assemble all the layers in a parallel ESN model.

    Args:
        units (int): Number of units in the reservoir.

        partitions (int): Number of partitions of the reservoir.

        overlap (int): Number of overlapping units between partitions.

        leak_rate (float): Leak rate of the reservoir.

        features (int): Number of features of the input data.

        activation (str): Activation function of the reservoir.

        input_reservoir_init (str): Initialization method for the input matrix.

        input_bias_init (str): Initialization method for the input bias.

        reservoir_kernel_init (str): Initialization method for the reservoir matrix.

        exponent (int): Exponent of the input matrix.

        seed (int): Seed for the random number generator.

    Return:
        model (ESN): Return the loaded instance of the ESN model.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000)
    print(f"\nSeed: {seed}\n")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    reservoir = parallel_esn(
        units=units,
        partitions=partitions,
        overlap=overlap,
        leak_rate=leak_rate,
        activation=activation,
        features=features,
        input_reservoir_init=input_reservoir_init,
        input_bias_init=input_bias_init,
        reservoir_kernel_init=reservoir_kernel_init,
        exponent=exponent,
    )

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = ESN(reservoir=reservoir, readout=readout_layer, seed=seed)
    return model


def generate_ECA_ESN(
    units: int,
    rule: Union[str, int, np.ndarray, list, tf.Tensor] = 110,
    steps: int = 1,
    leak_rate: float = 1.0,
    features: int = 1,
    activation: str = "tanh",
    input_reservoir_init: str = "InputMatrix",
    input_bias_init: str = "random_uniform",
    exponent: int = 2,
    seed: int = None,
) -> ESN:
    """
    Assemble all the layers in an ECA ESN model.

    Args:
        units (int): Number of units in the reservoir.

        rule (Union[str, int, np.ndarray, list, tf.Tensor]): The rule to use for the ECA reservoir.

        steps (int): Number of steps to run the ECA reservoir.

        leak_rate (float): Leak rate of the reservoir.

        features (int): Number of features of the input data.

        activation (str): Activation function of the reservoir.

        input_reservoir_init (str): Initialization method for the input matrix.

        input_bias_init (str): Initialization method for the input bias.

        exponent (int): Exponent of the input matrix.

        seed (int): Seed for the random number generator.

    Return:
        model (ESN): Return the loaded instance of the ESN model.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000)
    print(f"\nSeed: {seed}\n")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    reservoir = eca_esn(
        units=units,
        rule=rule,
        steps=steps,
        leak_rate=leak_rate,
        activation=activation,
        features=features,
        input_reservoir_init=input_reservoir_init,
        input_bias_init=input_bias_init,
        exponent=exponent,
    )

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = ESN(reservoir, readout_layer)
    return model


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
