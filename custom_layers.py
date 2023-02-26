"""Custom keras layers."""
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras

from custom_initializers import ErdosRenyi, InputMatrix

###############################################
################## Layers #####################
###############################################

# TODO: Add the input to the ESN cell remember what you call now input is feedback from the outputs


@tf.keras.utils.register_keras_serializable(package="custom")
class EsnCell(keras.layers.Layer):
    """Generates an ESN cell with the given parameters.

    To be used as the cell of a keras RNN.

    Args:
        units: Number of neurons in the reservoir. Default: 100.

        activation: Activation function to use. Can be a string or a function.

        leak_rate: A scalar between 0 and 1. Leak rate of the reservoir.

        input_initializer: Initializer for the input matrix.
            By default an InputMatrix.

        input_bias_initializer: Initializer for the input bias.
            By default a random uniform initializer.

        reservoir_initializer: Initializer for the reservoir matrix.
            By default an ErdosRenyi.

    Return:
        keras.layers.Layer:  A keras layer that can be used as a cell of a keras RNN.

    #### Example usage:
    >>> EsnCell = EsnCell(units=100,
                    activation='tanh',
                    leak_rate=1, input_initializer=input_initializer,
                    reservoir_initializer=reservoir_initializer,
                    input_bias_initializer=bias_initializer)
    >>> ESN = keras.layers.RNN(EsnCell, return_sequences=True)
    >>> output = ESN(input)
    >>> RNN_layer = keras.layers.RNN(EsnCell, return_sequences=True)
    """

    def __init__(
        self,
        units=100,
        activation="tanh",
        leak_rate=1,
        input_initializer=InputMatrix(),
        input_bias_initializer=keras.initializers.random_uniform(),
        reservoir_initializer=ErdosRenyi(),
        **kwargs,
    ) -> None:
        """Initialize the ESN cell."""
        self.input_initializer = input_initializer
        self.input_bias_initializer = input_bias_initializer

        self.reservoir_initializer = reservoir_initializer

        self.units = units
        self.activation = keras.activations.get(activation)

        # leak_rate integration. If leak_rate = 1, no leak_rate integration
        self.leak_rate = leak_rate

        # This property is required by keras. Keras will manage the states automatically.
        self.state_size = self.units
        self.input_dim = None

        # Initialize the weights
        self.w_input = None
        self.input_bias = None

        self.w_recurrent = None
        # self.reservoir_bias = None

        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        """
        Build the ESN cell.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.input_dim = input_shape[-1]

        # Input to reservoir matrix
        self.w_input = self.add_weight(
            name="input_to_Reservoir",
            shape=(self.input_dim, self.units),
            initializer=self.input_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        # Input bias
        self.input_bias = self.add_weight(
            name="input_bias",
            shape=(self.units,),
            initializer=self.input_bias_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        # Recurrent Matrix
        self.w_recurrent = self.add_weight(
            name="reservoir_kernel",
            shape=(self.units, self.units),
            initializer=self.reservoir_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs, states) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Combine the input and the states into a single output.

        Args:
            inputs (tf.Tensor): The input to the cell.

            states ([tf.Tensor]): The hidden state to the cell.
        Returns:
            output (tf.Tensor): The output of the cell.

            new_states ([tf.Tensor]): The new hidden state of the cell.
        """
        prev_output = states[0]

        # The input term.
        input_part = keras.backend.dot(inputs, self.w_input) + self.input_bias

        # The recurrent term.
        state_part = keras.backend.dot(prev_output, self.w_recurrent)

        # Producing the new state
        new_state = self.activation(input_part + state_part)

        # leak_rate integration
        output = (
            prev_output * (1 - self.leak_rate) + new_state * self.leak_rate
        )

        return output, [output]

    def get_config(self) -> Dict:
        """Get the config dictionary of the layer for serialization."""
        config = {
            "units": self.units,
            "activation": self.activation,
            "leak_rate": self.leak_rate,
            "input_initializer": self.input_initializer,
            "input_bias_initializer": self.input_bias_initializer,
            "reservoir_initializer": self.reservoir_initializer,
        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="custom")
class PowerIndex(keras.layers.Layer):
    """Applies a power function to the input even/odd indexed elements.

    Index can be an integer, depending on its parity will power the
    corresponding elements of the same parity from the input.

    Args:
        index (int): The index of the power function.

        exponent (float): The exponent of the power function.

    Returns:
        keras.layers.Layer: A keras layer that applies a power function to the
            input elements of the same parity as index.

    #### Example usage:

    >>> layer = PowerIndex(index=2, exponent=2)
    >>> layer(tf.constant([1, 2, 3, 4]))
    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 1,  4,  3, 16], dtype=int32)>
    """

    def __init__(self, index, exponent, **kwargs) -> None:
        """Initialize the layer."""
        self.index = (index) % 2
        self.exponent = exponent
        super().__init__(**kwargs)

    def call(self, inputs) -> tf.Tensor:
        """Compute the output tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor with the elements of the same
                parity of index powered by exponent.
        """
        dim = tf.shape(inputs)[-1]

        mask = tf.math.mod(tf.range(dim), 2)

        if self.index:
            mask = 1 - mask

        masked = tf.math.multiply(tf.cast(mask, tf.float32), inputs)

        unmaksed = tf.math.multiply(1 - tf.cast(mask, tf.float32), inputs)

        output = tf.math.pow(masked, self.exponent) + unmaksed

        return output

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """Compute the output shape.

        Args:
            input_shape (tf.TensorShape): Input shape.

        Returns:
            tf.TensorShape: Output shape same as input shape.
        """
        return tf.TensorShape(input_shape)

    def get_config(self) -> Dict:
        """Get the config dictionary of the layer for serialization."""
        config = {"index": self.index, "exponent": self.exponent}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_weights(self) -> List:
        """Return the weights of the layer."""
        return []


# For the ParallelReservoir model
@tf.keras.utils.register_keras_serializable(package="custom")
class InputSplitter(keras.layers.Layer):
    """Splits the input tensor into partitions with overlap on both sides."""

    def __init__(self, partitions, overlap, **kwargs) -> None:
        """Initialize the layer."""
        self.partitions = partitions
        self.overlap = overlap
        super().__init__(**kwargs)

    def call(self, inputs) -> tf.Tensor:
        """Compute the output tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, T, D)

        Returns:
            tf.Tensor: Tensor of shape (partitions, batch_size, T, D/partitions + 2*overlap)
        """
        if self.partitions == 1:
            return inputs.reshape(1, *inputs.shape)

        features = inputs.shape[-1]  # Change this to a[-1] later

        assert (
            features % self.partitions == 0
        ), "Input length must be divisible by partitions"

        input_clusters = [0 for _ in range(self.partitions)]

        # input_clusters = tf.Variable(input_clusters, dtype=tf.float32)

        # First roll the input tensor to the right by overlap
        inputs = tf.roll(inputs, self.overlap, axis=-1)

        for i in range(self.partitions):
            # Take into account the overlap on both sides is guaranteed
            # since we rolled the input tensor to the right by overlap before
            slicee = inputs[
                :, :, : features // self.partitions + 2 * self.overlap
            ]

            # slicee = slicee[0]

            # print(input_clusters[i].shape)
            # print(input_clusters.shape)

            input_clusters[i] = slicee

            # Just roll the input tensor to the left by N/partitions,
            # this will take care of the overlap on the left side
            inputs = tf.roll(
                inputs, shift=-features // self.partitions, axis=-1
            )

        input_clusters = tf.convert_to_tensor(input_clusters)

        return input_clusters

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """Compute the output shape.

        Args:
            input_shape (tf.TensorShape): Input shape.

        Returns:
            tf.TensorShape: Output shape same as input shape.
        """
        batches = input_shape[0]
        timesteps = input_shape[1]
        features = input_shape[2]

        shape = (
            self.partitions,
            batches,
            timesteps,
            features // self.partitions + 2 * self.overlap,
        )

        return tf.TensorShape(shape)

    def get_config(self) -> Dict:
        """Get the config dictionary of the layer for serialization."""
        config = {"partitions": self.partitions, "overlap": self.overlap}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_weights(self) -> List:
        """Return the weights of the layer."""
        return []


@tf.keras.utils.register_keras_serializable(package="custom")
class ReservoirCell(keras.layers.Layer):
    """Calculates the next internal states attending to reservoir_function.

    Args:
        reservoir_function: A reservoir that determines the internal dynamics of the state updates.
            Can be an oscillator or a cellular automaton, or any other complex system that
            takes a state and returns a new state.

        input_initializer: Initializer for the input weights.

        input_bias_initializer: Initializer for the input bias.

        activation: Activation function to use.

        leak_rate: Leak rate of the reservoir.
    """

    def __init__(
        self,
        reservoir_function,
        input_initializer=InputMatrix(),
        input_bias_initializer=keras.initializers.random_uniform(),
        activation="tanh",
        leak_rate=1,
        **kwargs,
    ) -> None:
        """Initialize the layer."""
        # Initialize the Reservoir
        self.reservoir_function = tf.function(
            reservoir_function
        )  # WARNING: This is experimental

        self.input_initializer = input_initializer
        self.input_bias_initializer = input_bias_initializer

        self.activation = keras.activations.get(activation)

        self.leak_rate = leak_rate

        # Initialize the weights
        self.w_input = None
        self.input_bias = None

        super().__init__(self, **kwargs)

    def build(self, input_shape) -> None:
        """Build the reservoir.

        Args:
            input_shape (tf.TensorShape): Input shape.
        """

        # Input to reservoir matrix
        self.w_input = self.add_weight(
            name="input_to_Reservoir",
            shape=(self.input_dim, self.units),
            initializer=self.input_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        # Input bias
        self.input_bias = self.add_weight(
            name="input_bias",
            shape=(self.units,),
            initializer=self.input_bias_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs, states) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Combine the input and the states into a single output.

        Args:
            inputs (tf.Tensor): The input to the cell.

            states ([tf.Tensor]): The hidden state to the cell.
        Returns:
            output (tf.Tensor): The output of the cell.

            new_states ([tf.Tensor]): The new hidden state of the cell.
        """
        prev_state = states[0]

        # The input term.
        input_part = keras.backend.dot(inputs, self.w_input) + self.input_bias

        # The reservoir term.
        state_part = self.reservoir_function(prev_state)

        new_state = self.activation(input_part + state_part)

        output = self.leak_rate * new_state + (1 - self.leak_rate) * prev_state

        return output, [output]

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        """Compute the output shape.

        Args:
            input_shape (tf.TensorShape): Input shape.

        Returns:
            tf.TensorShape: Output shape same as input shape.
        """
        return tf.TensorShape(input_shape)

    def get_config(self) -> Dict:
        """Get the config dictionary of the layer for serialization."""
        config = {
            "reservoir_function": self.reservoir_function,
            "input_initializer": self.input_initializer,
            "input_bias_initializer": self.input_bias_initializer,
            "activation": self.activation,
            "leak_rate": self.leak_rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_layers = {
    "EsnCell": EsnCell,
    "PowerIndex": PowerIndex,
    "InputSplitter": InputSplitter,
    "ReservoirCell": ReservoirCell,
}

keras.utils.get_custom_objects().update(custom_layers)
