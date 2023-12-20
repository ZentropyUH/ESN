"""Custom keras layers."""
from typing import Dict, List, Tuple

import tensorflow as tf
import keras
import keras.utils
import keras.layers
import keras.initializers
import keras.activations


from src.customs.custom_initializers import ErdosRenyi, InputMatrix
from src.customs.custom_reservoirs import create_automaton_tf

###############################################
################## Layers #####################
###############################################

# TODO: Add the input to the ESN cell remember what you call now input is feedback from the outputs


@keras.saving.register_keras_serializable(package="MyLayers", name="EsnCell")
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
        input_bias_initializer=keras.initializers.get("random_uniform"),
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
            shape=(1, self.units),
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
        input_part = keras.ops.dot(inputs, self.w_input) + self.input_bias

        # The recurrent term.
        state_part = keras.ops.dot(prev_output, self.w_recurrent)

        # Producing the new state
        new_state = self.activation(input_part + state_part)

        # leak_rate integration
        output = (
            prev_output * (1 - self.leak_rate) + new_state * self.leak_rate        )

        return output, [output]

    def get_config(self) -> Dict:
        """Get the config dictionary of the layer for serialization."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation.__name__,
                "leak_rate": self.leak_rate,
                "input_initializer": self.input_initializer,
                "input_bias_initializer": self.input_bias_initializer,
                "reservoir_initializer": self.reservoir_initializer,
            }
        )
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


@keras.saving.register_keras_serializable(package="MyLayers", name="PowerIndex")
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
        config = super().get_config()
        config.update({"index": self.index, "exponent": self.exponent})
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

    def get_weights(self) -> List:
        """Return the weights of the layer."""
        return []


# For the ParallelReservoir model
@keras.saving.register_keras_serializable(package="MyLayers", name="InputSplitter")
class InputSplitter(keras.layers.Layer):
    def __init__(self, partitions, overlap, **kwargs):
        super(InputSplitter, self).__init__(**kwargs)
        self.partitions = partitions
        self.overlap = overlap

    def call(self, inputs):
        # Handling the case when partitions are 1
        if self.partitions == 1:
            return [inputs]

        # Shape validation
        batch_size, sequence_length, features = inputs.shape
        assert features % self.partitions == 0, "Feature dimension must be divisible by partitions"
        assert features // self.partitions + 1 > self.overlap, "Overlap must be smaller than the length of the partitions."
        
        

        # Calculating the width of each partition including overlap
        partition_width = features // self.partitions + 2 * self.overlap

        # Applying circular wrapping
        wrapped_inputs = tf.concat([inputs[:, :, -self.overlap:], inputs, inputs[:, :, :self.overlap]], axis=-1)

        # Slicing the input tensor into partitions
        partitions = []
        for i in range(self.partitions):
            start = i * (features // self.partitions)
            end = start + partition_width
            partitions.append(wrapped_inputs[:, :, start:end])

        return partitions

    def compute_output_shape(self, input_shape):
        batch_size, sequence_length, features = input_shape
        partition_width = features // self.partitions + 2 * self.overlap
        return [(batch_size, sequence_length, partition_width) for _ in range(self.partitions)]

    def get_config(self):
        config = super(InputSplitter, self).get_config()
        config.update({
            'partitions': self.partitions,
            'overlap': self.overlap
        })
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


@keras.saving.register_keras_serializable(package="MyLayers", name="PseudoInverseRegression")
class PseudoInverseRegression(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(PseudoInverseRegression, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        super(PseudoInverseRegression, self).build(input_shape)

    def call(self, inputs):
        pseudo_inverse = tf.linalg.pinv(self.kernel)
        return tf.matmul(inputs, pseudo_inverse)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


@keras.saving.register_keras_serializable(package="MyLayers", name="ReservoirCell")
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
        units=100,
        input_initializer=InputMatrix(),
        input_bias_initializer=keras.initializers.get("random_uniform"),
        activation="tanh",
        leak_rate=1,
        **kwargs,
    ) -> None:
        """Initialize the layer."""
        # Initialize the Reservoir
        self.input_initializer = input_initializer
        self.input_bias_initializer = input_bias_initializer
        
        self.reservoir_function = reservoir_function

        self.units = units
        self.activation = keras.activations.get(activation)
        
        self.leak_rate = leak_rate
        
        self.state_size = self.units
        self.input_dim = None

        # Initialize the weights
        self.w_input = None
        self.input_bias = None

        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        """Build the reservoir.

        Args:
            input_shape (tf.TensorShape): Input shape.
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
            shape=(
                1,
                self.units,
            ),
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
        input_part = keras.ops.dot(inputs, self.w_input) + self.input_bias

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
        config = super().get_config()
        config.update(
            {
                "reservoir_function": self.reservoir_function,
                "input_initializer": self.input_initializer,
                "input_bias_initializer": self.input_bias_initializer,
                "activation": self.activation.__name__,
                "leak_rate": self.leak_rate,
            }
        )
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

@keras.saving.register_keras_serializable(package="MyLayers", name="ECACell")
class AutomatonCell(tf.keras.layers.Layer):
    def __init__(self, rule: Union[int, str, np.ndarray, tf.Tensor], **kwargs):
        super(AutomatonCell, self).__init__(**kwargs)
        self.rule = self._process_rule(rule)

    def _process_rule(self, rule):
        if isinstance(rule, int):
            rule = np.array([i for i in "{0:08b}".format(rule)], dtype=float)
        elif isinstance(rule, str):
            rule = np.array([i for i in rule], dtype=float)

        rule = np.flip(rule, axis=[0])
        num_neighbors = int((np.log2(len(rule)) - 1) / 2)
        assert num_neighbors == ((np.log2(len(rule)) - 1) / 2), "Rule length must be 2^n"
        return tf.convert_to_tensor(rule, dtype=tf.float32)

    def call(self, inputs, states):
        state_vector = states[0]
        new_state = self.automaton(state_vector)
        return new_state, [new_state]

    def automaton(self, state_vector):
        # Implement the automaton logic here, similar to your provided automaton function
        # ...

        # Returning the updated state_vector as an example
        return state_vector
    
    

def simple_esn(units: int, 
               leak_rate: float = 1, 
               features: int = 1,
               activation: str = 'tanh',
               input_reservoir_init: str = "InputMatrix",
               input_bias_init: str = "random_uniform",
               reservoir_kernel_init: str = "WattsStrogatzNX",
               exponent: int = 2
):
    
    inputs = keras.Input(batch_shape=(1, None, features), name='Input')
    
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
    
    power_index = PowerIndex(exponent=exponent, index=2, name="pwr")(
        esn_rnn
    )
    
    output = keras.layers.Concatenate(name="Concat_ESN_input")(
        [inputs, power_index]
    )
    
    reservoir = keras.Model(
        inputs=inputs,
        outputs=output,
    )
    
    return reservoir

def parallel_esn(units: int, 
                 leak_rate: float = 1, 
                 features: int = 1,
                 activation: str = 'tanh',
                 input_reservoir_init: str = "InputMatrix",
                 input_bias_init: str = "random_uniform",
                 reservoir_kernel_init: str = "WattsStrogatzNX",
                 exponent: int = 2,
                 partitions: int = 1,
                 overlap: int = 0
):
    
    # FIX
    assert features % partitions == 0, "Input length must be divisible by partitions"
    
    assert features // partitions > overlap, "Overlap must be smaller than the length of the partitions"
    
    inputs = keras.Input(batch_shape=(1, None, features), name='Input')
        
    inputs_splitted = InputSplitter(partitions=partitions, overlap=overlap, name="splitter")(inputs)
    
    
    # Create the reservoirs
    reservoir_outputs = []
    for i in range(partitions):
        
        esn_cell = EsnCell(
            units=units,
            name="EsnCell",
            activation=activation,
            leak_rate=leak_rate,
            input_initializer=input_reservoir_init,
            input_bias_initializer=input_bias_init,
            reservoir_initializer=reservoir_kernel_init,
        )

        reservoir = keras.layers.RNN(
                esn_cell,
                trainable=False,
                stateful=True,
                return_sequences=True,
                name=f"esn_rnn_{i}",
            )
                
        reservoir_output = reservoir(inputs_splitted[i])
        reservoir_output = PowerIndex(exponent=exponent, index=i, name=f"pwr_{i}")(reservoir_output)
        reservoir_outputs.append(reservoir_output)
    
    
    # Concatenate the power indices
    output = keras.layers.Concatenate(name="esn_rnn")(reservoir_outputs)
    
    output = keras.layers.Concatenate(name="Concat_ESN_input")([inputs, output])
    
    parallel_reservoir = keras.Model(
        inputs=inputs,
        outputs=output,
    )
    
    return parallel_reservoir

def eca_esn(units: int,
            leak_rate: float = 1,
            features: int = 1,
            activation: str = 'tanh',
            input_reservoir_init: str = "InputMatrix",
            input_bias_init: str = "random_uniform",
            rule: int = 110,
            steps: int = 1,
            exponent: int = 2,
):
    
    eca_function = create_automaton_tf(rule, steps=steps)
    
    inputs = keras.Input(batch_shape=(1, None, features), name='Input')
    
    eca_cell = ReservoirCell(
        units=units,
        reservoir_function=eca_function,
        input_initializer=input_reservoir_init,
        input_bias_initializer=input_bias_init,
        activation=activation,
        leak_rate=leak_rate,
        name="EcaCell",
    )
    
    eca_rnn = keras.layers.RNN(
        eca_cell,
        trainable=False,
        stateful=True,
        return_sequences=True,
        name="esn_rnn",
    )(inputs)
    
    power_index = PowerIndex(exponent=exponent, index=2, name="pwr")(
        eca_rnn
    )
    
    output = keras.layers.Concatenate(name="Concat_ESN_input")(
        [inputs, power_index]
    )
    
    reservoir = keras.Model(
        inputs=inputs,
        outputs=output,
    )
    
    return reservoir


# custom_layers = {
#     "EsnCell": EsnCell,
#     "PowerIndex": PowerIndex,
#     "InputSplitter": InputSplitter,
#     "ReservoirCell": ReservoirCell,
# }

# keras.utils.get_custom_objects().update(custom_layers)
