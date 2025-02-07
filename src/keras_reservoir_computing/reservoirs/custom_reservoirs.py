"""Custom Reservoirs for Keras."""
from abc import ABC, abstractmethod
from typing import Callable, Union

import keras

from keras_reservoir_computing.layers import PowerIndex


@keras.saving.register_keras_serializable(package="Reservoirs", name="BaseReservoirCell")
class BaseReservoirCell(keras.layers.Layer, ABC):
    """
    Abstract base class for different types of reservoir cells. This is the one-step computation unit of a reservoir.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    @abstractmethod
    def build(self, input_shape):
        pass

    @abstractmethod
    def call(self, inputs, states):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
                        "units": self.units,
                    })
        return config


@keras.saving.register_keras_serializable(package="Reservoirs", name="BaseReservoir")
class BaseReservoir(keras.Model, ABC):
    """
    Abstract base class for different types of reservoirs.
    All reservoirs should inherit from this class. This wraps the reservoir cell within an RNN layer.

    Args:
        reservoir_cell (BaseReservoirCell): The reservoir cell to use in the reservoir.

    Returns:
        keras.layers.Layer: The reservoir layer.
    """
    def __init__(self, reservoir_cell, **kwargs):
        super().__init__(**kwargs)

        self.reservoir_cell = reservoir_cell

    def build(self, input_shape):
        if self.built:
            return

        self.rnn_layer = keras.layers.RNN(
            self.reservoir_cell,
            trainable=False,
            stateful=True,
            return_sequences=True,
            name="reservoir_rnn"
            )

        self.rnn_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = self.rnn_layer(inputs, **kwargs)
        return output

    def get_states(self):
        return self.rnn_layer.states

    def set_states(self, new_states):
        """
        Sets the new states of the Reservoir (i. e., the rnn_layer within)

        Args:
            new_states (List): The list of new states of the reservoir. Should maintain the shapes of the internal states.
        """
        if not self.built:
            raise RuntimeError("The Reservoir must be built first.")

        if len(self.rnn_layer.states) != len(new_states):
            raise ValueError(f"The new states are of length {len(new_states)}, must be of same size and shapes as the Reservoir States {len(self.rnn_layer.states)}.")
        else:
            for (i, new_state) in enumerate(new_states):
                state_shape = self.rnn_layer.states[i].shape
                if state_shape != new_state.shape:
                    raise ValueError(f"The {i}th new state is of shape {new_state.shape}, should be of shape {state_shape}.")
                else:
                    self.rnn_layer.states[i].assign(new_state)

    def reset_states(self):
        self.rnn_layer.reset_states()

    @property
    def units(self):
        return self.reservoir_cell.units

    @abstractmethod
    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
                        "reservoir_cell": keras.layers.serialize(self.reservoir_cell),
                    })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize reservoir_cell from config
        reservoir_cell_config = config.pop('reservoir_cell')
        reservoir_cell = keras.layers.deserialize(reservoir_cell_config)

        # Return the instance of BaseReservoir with deserialized reservoir_cell
        return cls(reservoir_cell=reservoir_cell, **config)

# Classic ESN components, TODO: Implement the cell that receives an input drive signal as well as the feedback.

@keras.saving.register_keras_serializable(package="Reservoirs", name="ESNCell")
class ESNCell(BaseReservoirCell):
    """
    Simple Reservoir cell implementing the classic Echo State Network (ESN) model.
    """
    def __init__(
        self,
        units: int,
        leak_rate: float = 1.0,
        noise_level: float = 0.0, # Add noise to the reservoir state only during training
        activation: Union[str, Callable] = "tanh",
        input_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
        input_bias_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
        **kwargs,
    ):
        super().__init__(units, **kwargs)
        self.leak_rate = leak_rate
        self.noise_level = noise_level
        self.activation = keras.activations.get(activation)
        self.input_initializer = keras.initializers.get(input_initializer)
        self.input_bias_initializer = keras.initializers.get(input_bias_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):

        features = input_shape[-1]

        self.input_kernel = self.add_weight(
            shape=(features, self.units),
            initializer=self.input_initializer,
            trainable=False,
        )
        self.input_bias = self.add_weight(
            shape=(1, self.units,),
            initializer=self.input_bias_initializer,
            trainable=False,
        )
        self.reservoir_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs, states, training=False):
        """
        Compute one step of the reservoir according to the ESN model. With leaky integration and noise if specified.

        Args:
            inputs (tf.Tensor): The input tensor.
            states (List): The list of states of the reservoir.
            training (bool): Whether the model is in training mode or not.

        Returns:
            tf.Tensor: The new state of the reservoir.
        """
        prev_state = states[0]

        input_part = keras.ops.matmul(inputs, self.input_kernel) + self.input_bias

        state_part = keras.ops.matmul(prev_state, self.reservoir_kernel)

        output = self.activation(input_part + state_part)

        # Add noise to the reservoir state only during training
        if training:
            output = output + self.noise_level * keras.random.normal(keras.backend.shape(output))

        # Leaky integration
        # The casting of 1 is a crazy thing when operating with python floats and tf or np floats, it promotes the operation to the highest precision, i.e. tf.float64, yet we are using tf.float32. What in the actual fuck?
        lag = prev_state * (keras.ops.cast(1, keras.backend.floatx()) - self.leak_rate)
        update = output * self.leak_rate

        new_state = lag + update

        return new_state, [new_state]

    def get_config(self):
        """
        Get the configuration of the ESNCell.

        Returns:
            dict: The configuration of the ESNCell.
        """
        config = super().get_config()
        config.update({
                        "leak_rate": self.leak_rate,
                        "noise_level": self.noise_level,
                        "activation": keras.activations.serialize(self.activation),
                        "input_initializer": keras.initializers.serialize(self.input_initializer),
                        "input_bias_initializer": keras.initializers.serialize(self.input_bias_initializer),
                        "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                    })

        return config


@keras.saving.register_keras_serializable(package="Reservoirs", name="EchoStateNetwork")
class EchoStateNetwork(BaseReservoir):
    """
    Simple Reservoir model implementing the classic Echo State Network (EchoStateNetwork).
    """
    def __init__(
        self,
        index: int = 2, # Index parity for power_index augmentation
        exponent: int = 2, # For the power_index state augmentation
        **kwargs,
    ):
        super().__init__(**kwargs) # It will handle the cell here

        self.index = index
        self.exponent = exponent
        self.power_index = PowerIndex(exponent=exponent, index=2, name="pwr")
        self.concatenate = keras.layers.Concatenate(name="Concat_ESN_input")

    def call(self, inputs, **kwargs):
        states = self.rnn_layer(inputs, **kwargs)
        power_index = self.power_index(states, **kwargs)
        output = self.concatenate([inputs, power_index], **kwargs)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
                        "index": self.index,
                        "exponent": self.exponent,
                    })

        return config

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        return input_shape[:-1] + (input_shape[-1] + self.reservoir_cell.units,)
