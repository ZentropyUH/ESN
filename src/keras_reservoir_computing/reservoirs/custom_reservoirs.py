"""
Custom Reservoirs for Keras.

This module defines abstract base classes and concrete implementations
for custom reservoir layers and cells in Keras, following the Echo
State Network (ESN) paradigm.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union, List, Tuple

import tensorflow as tf
import keras
from keras.src.initializers import Initializer

from keras_reservoir_computing.layers import PowerIndex


@keras.saving.register_keras_serializable(
    package="Reservoirs", name="BaseReservoirCell"
)
class BaseReservoirCell(keras.layers.Layer, ABC):
    """
    Abstract base class for different types of reservoir cells.
    Each reservoir cell represents the one-step computation unit
    of a reservoir (akin to an RNN cell).

    Parameters
    ----------
    units : int
        Number of units in the reservoir cell.

    Attributes
    ----------
    units : int
        Number of units in the reservoir cell.
    state_size : int
        Size of the state (same as `units`).
    """
    def __init__(self, units: int, **kwargs) -> None:
        """
        Initialize the BaseReservoirCell.

        Parameters
        ----------
        units : int
            Number of units in the reservoir cell.
        **kwargs : dict
            Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    @abstractmethod
    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Create the weights of the reservoir cell.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs.
        """
        pass

    @abstractmethod
    def call(
        self, inputs: tf.Tensor, states: List[tf.Tensor], training: bool = False
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Forward pass of one time step of the reservoir cell.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor for the current time step.
        states : List[tf.Tensor]
            Previous state(s) of the reservoir.
        training : bool, optional
            Whether the call is in training mode, by default False

        Returns
        -------
        tf.Tensor
            The new output state of the reservoir cell.
        List[tf.Tensor]
            A list containing the new state(s).
        """
        pass

    def get_config(self) -> dict:
        """
        Return the configuration of the BaseReservoirCell.

        Returns
        -------
        dict
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="Reservoirs", name="BaseReservoir")
class BaseReservoir(keras.Model, ABC):
    """
    Abstract base class for different types of reservoirs.
    This wraps the reservoir cell within a Keras RNN layer.

    Parameters
    ----------
    reservoir_cell : BaseReservoirCell
        The reservoir cell to use in the reservoir.

    Attributes
    ----------
    reservoir_cell : BaseReservoirCell
        The reservoir cell used inside this Reservoir.
    rnn_layer : keras.layers.RNN
        The RNN layer that internally wraps the reservoir cell.
    """
    def __init__(self, reservoir_cell: BaseReservoirCell, **kwargs) -> None:
        """
        Initialize the BaseReservoir.

        Parameters
        ----------
        reservoir_cell : BaseReservoirCell
            The reservoir cell to be used.
        **kwargs : dict
            Additional keyword arguments for the Model base class.
        """
        super().__init__(**kwargs)
        self.reservoir_cell = reservoir_cell
        self.rnn_layer: keras.layers.RNN = None  # Will be built later in build()

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Create the internal RNN layer using the given reservoir cell.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs.
        """
        if self.built:
            return

        self.rnn_layer = keras.layers.RNN(
            self.reservoir_cell,
            trainable=False,  # Classic ESN uses a fixed reservoir
            stateful=True,
            return_sequences=True,
            name="reservoir_rnn",
        )
        self.rnn_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Forward pass of the reservoir.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, features).
        **kwargs : dict
            Additional keyword arguments for the RNN call.

        Returns
        -------
        tf.Tensor
            Output of the reservoir (the RNN layer).
        """
        return self.rnn_layer(inputs, **kwargs)

    def get_states(self) -> List[tf.Tensor]:
        """
        Retrieve the current states of the reservoir's RNN layer.

        Returns
        -------
        List[tf.Tensor]
            The current state(s) of the reservoir.
        """
        return self.rnn_layer.states

    def set_states(self, new_states: List[tf.Tensor]) -> None:
        """
        Set the reservoir's RNN layer states.

        Parameters
        ----------
        new_states : List[tf.Tensor]
            A list of new states for the reservoir.

        Raises
        ------
        RuntimeError
            If the reservoir is not yet built.
        ValueError
            If the length or shapes of `new_states` do not match the existing states.
        """
        if not self.built:
            raise RuntimeError("The Reservoir must be built first.")

        if len(self.rnn_layer.states) != len(new_states):
            raise ValueError(
                f"The new states are of length {len(new_states)}, must match "
                f"the Reservoir's states length {len(self.rnn_layer.states)}."
            )

        for i, new_state in enumerate(new_states):
            expected_shape = self.rnn_layer.states[i].shape
            if expected_shape != new_state.shape:
                raise ValueError(
                    f"The {i}th new state is of shape {new_state.shape}, "
                    f"should be {expected_shape}."
                )
            self.rnn_layer.states[i].assign(new_state)

    def reset_states(self) -> None:
        """
        Reset the reservoir's RNN layer states to zeros.
        """
        self.rnn_layer.reset_states()

    @property
    def units(self) -> int:
        """
        Number of units in the underlying reservoir cell.

        Returns
        -------
        int
            The number of units of the reservoir cell.
        """
        return self.reservoir_cell.units

    @abstractmethod
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Compute the output shape of the reservoir.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs.

        Returns
        -------
        tf.TensorShape
            The output shape of the reservoir.
        """
        pass

    def get_config(self) -> dict:
        """
        Return the configuration of the BaseReservoir.

        Returns
        -------
        dict
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update(
            {
                "reservoir_cell": keras.layers.serialize(self.reservoir_cell),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Create an instance of the reservoir from a config dictionary.

        Parameters
        ----------
        config : dict
            The config dictionary.

        Returns
        -------
        BaseReservoir
            An instance of the reservoir.
        """
        reservoir_cell_config = config.pop("reservoir_cell")
        reservoir_cell = keras.layers.deserialize(reservoir_cell_config)
        return cls(reservoir_cell=reservoir_cell, **config)


@keras.saving.register_keras_serializable(package="Reservoirs", name="ESNCell")
class ESNCell(BaseReservoirCell):
    """
    Simple Reservoir cell implementing the classic Echo State Network (ESN) model.

    Parameters
    ----------
    units : int
        Number of units in the reservoir cell.
    leak_rate : float, optional
        Leaky integration rate, by default 1.0
    noise_level : float, optional
        Standard deviation of noise added to the state (only during training), by default 0.0
    activation : str or Callable, optional
        Activation function, by default 'tanh'
    input_initializer : str or Initializer, optional
        Initializer for input weights, by default 'random_uniform'
    input_bias_initializer : str or Initializer, optional
        Initializer for input bias, by default 'random_uniform'
    kernel_initializer : str or Initializer, optional
        Initializer for the reservoir recurrent weights, by default 'random_uniform'

    Attributes
    ----------
    leak_rate : float
        Leaky integration rate.
    noise_level : float
        Standard deviation of noise added to the state.
    activation : Callable
        Activation function used in the reservoir.
    input_initializer : Initializer
        Initializer for input weights.
    input_bias_initializer : Initializer
        Initializer for input bias.
    kernel_initializer : Initializer
        Initializer for reservoir recurrent weights.
    """
    def __init__(
        self,
        units: int,
        leak_rate: float = 1.0,
        noise_level: float = 0.0,
        activation: Union[str, Callable] = "tanh",
        input_initializer: Union[str, Initializer] = "random_uniform",
        input_bias_initializer: Union[str, Initializer] = "random_uniform",
        kernel_initializer: Union[str, Initializer] = "random_uniform",
        **kwargs,
    ) -> None:
        super().__init__(units, **kwargs)
        self.leak_rate = leak_rate
        self.noise_level = noise_level
        self.activation = keras.activations.get(activation)
        self.input_initializer = keras.initializers.get(input_initializer)
        self.input_bias_initializer = keras.initializers.get(input_bias_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Create the input, bias, and reservoir kernels.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs (batch, timesteps, features).
        """
        features = input_shape[-1]

        self.input_kernel = self.add_weight(
            shape=(features, self.units),
            initializer=self.input_initializer,
            trainable=False,
            name="input_kernel",
        )
        self.input_bias = self.add_weight(
            shape=(1, self.units),
            initializer=self.input_bias_initializer,
            trainable=False,
            name="input_bias",
        )
        self.reservoir_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            trainable=False,
            name="reservoir_kernel",
        )
        super().build(input_shape)

    def call(
        self, inputs: tf.Tensor, states: List[tf.Tensor], training: bool = False
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Perform one step of the ESN cell update.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor for the current time step (batch, features).
        states : List[tf.Tensor]
            List containing the previous reservoir state (batch, units).
        training : bool, optional
            Whether the layer is in training mode, by default False

        Returns
        -------
        tf.Tensor
            The new reservoir state of shape (batch, units).
        List[tf.Tensor]
            A list containing the new reservoir state (same as first return).
        """
        prev_state = states[0]

        # Linear transformation of inputs
        input_part = keras.ops.matmul(inputs, self.input_kernel) + self.input_bias
        # Recurrent part from previous state
        state_part = keras.ops.matmul(prev_state, self.reservoir_kernel)

        # Activation
        output = self.activation(input_part + state_part)

        # Add noise only in training mode
        if training:
            output += self.noise_level * keras.random.normal(tf.shape(output))

        # Leaky integration
        # The cast ensures consistent float types.
        one = keras.ops.cast(1, tf.keras.backend.floatx())
        lag = prev_state * (one - self.leak_rate)
        update = output * self.leak_rate
        new_state = lag + update

        return new_state, [new_state]

    def get_config(self) -> dict:
        """
        Return the configuration of the ESNCell.

        Returns
        -------
        dict
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update(
            {
                "leak_rate": self.leak_rate,
                "noise_level": self.noise_level,
                "activation": keras.activations.serialize(self.activation),
                "input_initializer": keras.initializers.serialize(
                    self.input_initializer
                ),
                "input_bias_initializer": keras.initializers.serialize(
                    self.input_bias_initializer
                ),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


@keras.saving.register_keras_serializable(package="Reservoirs", name="EchoStateNetwork")
class EchoStateNetwork(BaseReservoir):
    """
    Simple Reservoir model implementing the classic Echo State Network (ESN).
    This class augments the standard RNN output with a "power index" transformation
    before concatenating back to the original input.

    Parameters
    ----------
    reservoir_cell : BaseReservoirCell
        The reservoir cell to use (e.g., ESNCell).
    index : int, optional
        The index parameter for power index augmentation, by default 2.
    exponent : int, optional
        The exponent for the power index augmentation, by default 2.

    Attributes
    ----------
    index : int
        The index parameter for power index augmentation.
    exponent : int
        The exponent for power index augmentation.
    power_index : PowerIndex
        The layer that applies the power-index transformation.
    concatenate : keras.layers.Concatenate
        Keras layer used to concatenate the original input with the power-indexed state.
    """
    def __init__(
        self,
        reservoir_cell: BaseReservoirCell,
        index: int = 2,
        exponent: int = 2,
        **kwargs,
    ) -> None:
        """
        Initialize the EchoStateNetwork.

        Parameters
        ----------
        reservoir_cell : BaseReservoirCell
            The reservoir cell to use (e.g., ESNCell).
        index : int, optional
            The index parameter for power index augmentation, by default 2.
        exponent : int, optional
            The exponent for the power index augmentation, by default 2.
        **kwargs : dict
            Additional keyword arguments for the BaseReservoir.
        """
        super().__init__(reservoir_cell=reservoir_cell, **kwargs)
        self.index = index
        self.exponent = exponent
        self.power_index = PowerIndex(
            exponent=self.exponent, index=self.index, name="pwr"
        )
        self.concatenate = keras.layers.Concatenate(name="Concat_ESN_input")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Forward pass of the EchoStateNetwork.

        1. Passes the input through the reservoir RNN layer.
        2. Applies the power index transformation on the RNN outputs.
        3. Concatenates the original input with the power index outputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, features).
        **kwargs : dict
            Additional arguments for the internal RNN or layer calls.

        Returns
        -------
        tf.Tensor
            Concatenated tensor of shape (batch, time, features + reservoir_units).
        """
        states = self.rnn_layer(inputs, **kwargs)
        power_index = self.power_index(states, **kwargs)
        return self.concatenate([inputs, power_index], **kwargs)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Compute the output shape of EchoStateNetwork.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs (batch, time, features).

        Returns
        -------
        tf.TensorShape
            (batch, time, features + reservoir_units)
        """
        # Expect shape: (batch, time, features)
        # Output shape is (batch, time, features + self.reservoir_cell.units)
        if isinstance(input_shape, (list, tuple)):
            input_shape = tf.TensorShape(input_shape)
        return tf.TensorShape(
            [
                input_shape[0],
                input_shape[1],
                input_shape[-1] + self.reservoir_cell.units,
            ]
        )

    def get_config(self) -> dict:
        """
        Return the configuration of the EchoStateNetwork.

        Returns
        -------
        dict
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update(
            {
                "index": self.index,
                "exponent": self.exponent,
            }
        )
        return config
