"""Custom keras models."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from custom_layers import EsnCell, PowerIndex, InputSplitter, ReservoirCell


###############################################
################## Models #####################
###############################################


@tf.keras.utils.register_keras_serializable(package="custom")
class ESN(keras.Model):
    """
    A simple ESN model.

    Args:
        units: Number of units in the reservoir

        leak_rate: Leaky rate of the reservoir

        input_reservoir_init: Initializer for the input to reservoir weights.
            Defaults to InputMatrix()

        input_bias_init: Initializer for the input bias.
            Defaults to RandomUniform()

        reservoir_kernel_init: Initializer for the reservoir weights.
            Defaults to ErdosRenyi()

        esn_activation: Activation function of the reservoir.
            Defaults to tanh.

        exponent: Exponent of the power function applied to the reservoir.
            Defaults to 2.

        seed: Seed of the model

        raw: If True the model will not concatenate the input with the reservoir output

        **kwargs: Keyword arguments for the keras.Model class

    Returns:
        A keras.Model object

    Example usage:

    >>> model = ESN(units=100,
                         leak_rate=0.5,
                         input_reservoir_init='glorot_uniform',
                         input_bias_init='zeros',
                         reservoir_kernel_init='ErdosRenyi',
                         reservoir_bias_init='zeros',
                         esn_activation='tanh',
                         exponent=2,
                         seed=42)
    """

    def __init__(
        self,
        # ESN related parameters
        units=200,
        leak_rate=1,
        input_reservoir_init="InputMatrix",
        input_bias_init="random_uniform",
        reservoir_kernel_init="ErdosRenyi",
        esn_activation="tanh",
        exponent=2,
        # Seed of the model
        seed=None,
        raw=False,
        **kwargs,
    ):
        """Initialize the model."""
        super().__init__(**kwargs)

        # Units of the ESN and output of the readout layer

        self.units = units
        self.inputshape = None

        self.leak_rate = leak_rate

        # ESN related parameters
        self.input_reservoir_init = input_reservoir_init
        self.input_bias_init = input_bias_init

        self.reservoir_kernel_init = reservoir_kernel_init

        self.esn_activation = esn_activation

        self.raw = raw

        # Exponent of the even states
        self.exponent = exponent

        # Optional seed of the model
        if seed is None:
            seed = np.random.randint(0, 1000000)

        self.seed = seed

        print()
        print(f"Seed: {seed}\n")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.inputshape = None

        ###############################################
        ################## Layers #####################
        ###############################################

        # Recurrent cell
        esn_cell = EsnCell(
            units=self.units,
            name="EsnCell",
            activation=self.esn_activation,
            leak_rate=self.leak_rate,
            input_initializer=self.input_reservoir_init,
            input_bias_initializer=self.input_bias_init,
            reservoir_initializer=self.reservoir_kernel_init,
        )

        # RNN layer
        self.esn = keras.layers.RNN(
            esn_cell,
            trainable=False,
            stateful=True,
            return_sequences=True,
            name="esn_rnn",
        )

        self.power_index = PowerIndex(
            exponent=self.exponent, index=2, name="power_index"
        )

        self.input_reservoir_concatenation = keras.layers.Concatenate(
            name="Concat_ESN_input"
        )

    def build(self, input_shape):
        """Build the model.

        Saves the input shape in inputshape.

        Args:
            input_shape (tf.TensorShape): Input shape of the model.
        """
        self.inputshape = (None, input_shape[-1])
        super().build(input_shape)

    # # # # # # Call Function

    def call(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, time_steps, input_dim)

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, time_steps, units)
        """
        output = self.esn(inputs)
        output = self.power_index(output)
        if self.raw is False:
            output = self.input_reservoir_concatenation([inputs, output])
        return output

    # # # # # # For plotting the model

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the model.

        Args:
            input_shape (tf.TensorShape): Input shape of the model.

        Returns:
            tf.TensorShape: Output shape of the model.
        """
        return tf.TensorShape(
            [input_shape[0], input_shape[1], self.units + input_shape[-1]]
        )

    def build_graph(self):
        """Return a dummy model to use for plotting the model graph."""
        assert self.built, "Model must be built before calling build_graph()"
        dummy = keras.Input(shape=self.inputshape, name="Input")
        return keras.Model(inputs=[dummy], outputs=self.call(dummy))

    # # # # # # Methods for ESN cell

    # Set the state of the ESN

    def set_esn_state(self, state):
        """Set the state of the ESN cell.

        Args:
            state (tf.Tensor): State tensor of shape (batch_size, units)
        """
        self.esn.reset_states(states=state)

    # Reset the state of the ESN
    def reset_esn_state(self):
        """Reset the state of the ESN cell to zeros."""
        self.esn.reset_states()

    # Get the state of the ESN
    def get_esn_state(self):
        """Return the state of the ESN cell.

        Returns:
            tf.Tensor: State tensor of shape (units)
        """
        # return a copy of the state, else it would be a reference
        if self.built:
            return tf.identity(self.esn.states[0])
        raise ValueError(
            "The model must be built before calling get_esn_state()"
        )

    # # # # # # Pretraining methods

    def verify_esp(self, transient_data, times=1, verbose=1):
        """Verify the Echo State Property of the ESN cell.

        Args:
            transient_data (tf.Tensor): Transient data of shape (batch_size, time_steps, input_dim)

            times (int, optional): Number of times to verify the ESP. Defaults to 1.

        Returns:
            bool: True if the ESP is verified, False otherwise
        """
        # TODO: Verifies the Echo State Property. This is a naive implementation,
        # it should be improved. Maybe use the ESP_index idea from Galicchio et. al. 2018

        if not self.built:
            print("Model is not built yet, building model...")
            self.build(transient_data.shape)

        # Save current
        current_state = self.get_esn_state()

        for i in range(times):
            if verbose > 0:
                print(f"Verifying ESP, iteration {i+1} of {times}")
            # Generate random states first run
            state_1 = tf.random.uniform((1, self.units))
            self.set_esn_state(state_1)
            self.predict(transient_data, verbose=0)
            first = self.get_esn_state()

            # Generate random states second run
            state_2 = tf.random.uniform((1, self.units))
            self.set_esn_state(state_2)
            self.predict(transient_data, verbose=0)
            second = self.get_esn_state()

            distance = tf.norm(first - second)
            if verbose > 1:
                print(f"Distance between states: {distance}")

            esp = distance < 1e-5

            if not esp:
                print(
                    f"ESP is not satisfied, increase transient size. Failed at iteration: {i}"
                )
                # Restore previous state
                self.set_esn_state(current_state)
                return esp

        print("ESP is satisfied\n")
        # Restore previous state
        self.set_esn_state(current_state)
        return esp

    # # # # # # Methods for saving/loading the model (serializing)

    def get_config(self):
        """Get the config dictionary of the model for serialization."""
        config = super().get_config().copy()

        config.update(
            {
                "units": self.units,
                "leak_rate": self.leak_rate,
                "input_reservoir_init": self.input_reservoir_init,
                "input_bias_init": self.input_bias_init,
                "reservoir_kernel_init": self.reservoir_kernel_init,
                "esn_activation": self.esn_activation,
                "exponent": self.exponent,
                "seed": self.seed,
                "raw": self.raw,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="custom")
class ParallelESN(keras.Model):
    """Parallel version of the ESN.

    This model uses several EsnCell to process the input timeseries in parallel.
    It receives a timeseries of shape (batch_size, time_steps, input_dim) and process
    the different input_dim timeseries in clusters of size input_dim/reservoir_amount + overlap.
    The overlap is used exploit spatial correlations between the different input_dim timeseries.
    The output of the model is a timeseries of shape (batch_size, time_steps, input_dim).

    Args:
        units_per_reservoir (int): Number of units in the ESN cell.

        reservoir_amount (int): Number of ESN cells in the parallel model.
            The reservoir amount must divide the input dimension.

        overlap (int): Number of elements to overlap between the clusters with its neighbours.

        leak_rate (float, optional): Leak rate of the ESN cell. Defaults to 0.3.

        input_reservoir_init (str, optional): Initializer for the input reservoir
            weights. Defaults to "glorot_uniform".

        input_bias_init (str, optional): Initializer for the input bias weights.
            Defaults to "zeros".

        reservoir_kernel_init (str, optional): Initializer for the reservoir kernel
            weights. Defaults to "glorot_uniform".

        reservoir_bias_init (str, optional): Initializer for the reservoir bias
            weights. Defaults to "zeros".

        esn_activation (str, optional): Activation function of the ESN cell.
            Defaults to "tanh".

        exponent (int, optional): Exponent of the power index. Defaults to 2.

        seed (int, optional): Seed for the random number generator. Defaults to 42.
    """

    def __init__(
        self,
        # ESN related parameters
        units_per_reservoir=300,
        reservoir_amount=1,
        overlap=0,
        leak_rate=1,
        input_reservoir_init="InputMatrix",
        input_bias_init="random_uniform",
        reservoir_kernel_init="ErdosRenyi",
        esn_activation="tanh",
        # Exponent of the power index (Augmented Hidden ESN state)
        exponent=2,
        # Seed of the model
        seed=None,
        **kwargs,
    ):
        """Initialize the model."""
        super().__init__(**kwargs)

        if reservoir_amount == 1:
            print(
                "Warning: Using a single reservoir is equivalent to using a normal ESN"
            )

        self.units_per_reservoir = units_per_reservoir
        self.reservoir_amount = reservoir_amount
        self.overlap = overlap

        self.leak_rate = leak_rate
        self.input_reservoir_init = input_reservoir_init
        self.input_bias_init = input_bias_init
        self.reservoir_kernel_init = reservoir_kernel_init
        self.esn_activation = esn_activation

        self.exponent = exponent

        # Optional seed of the model
        if seed is None:
            seed = np.random.randint(0, 1000000)

        self.seed = seed

        print()
        print(f"Parallel ESN seed: {seed}\n")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.inputshape = None

        ###############################################
        ################## Layers #####################
        ###############################################

        # Input split layer
        self.input_split = InputSplitter(self.reservoir_amount, self.overlap)

        # Create ESN models in parallel
        self.esn_layers = [
            ESN(
                units=self.units_per_reservoir,
                leak_rate=self.leak_rate,
                input_reservoir_init=self.input_reservoir_init,
                input_bias_init=self.input_bias_init,
                reservoir_kernel_init=self.reservoir_kernel_init,
                # reservoir_bias_init=self.reservoir_bias_init,
                esn_activation=self.esn_activation,
                exponent=self.exponent,
                raw=True,
                # The seeds are generated randomly but are determined by the seed of the model
            )
            for _ in range(self.reservoir_amount)
        ]

        # Concatenate the augmented hidden states of the ESN cells
        self.concat_esn_layer = keras.layers.Concatenate()

        # Concatenate the overall output of the ESN cells with the input
        self.concat_input_layer = keras.layers.Concatenate()

    def build(self, input_shape):
        """Build the model."""
        esn_input_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2] // self.reservoir_amount + 2 * self.overlap,
        )
        # Build the ESN layers
        for esn_layer in self.esn_layers:
            esn_layer.build(esn_input_shape)
        self.inputshape = (None, input_shape[-1])
        super().build(input_shape)

    # # # # # # Call Function

    def call(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input timeseries of shape (batch_size, time_steps, input_dim).

        Returns:
            tf.Tensor: Output timeseries of shape
            (batch_size, time_steps, input_dim + reservoir_amount * (units_per_reservoir + 2*overlap)).

        """
        # Make sure the reservoir amount divides the input dimension
        assert (
            inputs.shape[-1] % self.reservoir_amount == 0
        ), "The reservoir amount must divide the input features, i.e. the input dimension."

        output = self.input_layer(inputs)

        # Split the input into clusters of size input_dim/reservoir_amount + overlap
        # The overlap is used to exploit spatial correlations
        # between the different input_dim timeseries
        input_clusters = self.input_split(inputs)

        esn_outputs = []

        for i, esn_layer in enumerate(self.esn_layers):
            # Get the current input cluster
            input_cluster = input_clusters[i]

            # Get the output of the current ESN cell
            cluster_output = esn_layer(input_cluster)

            esn_outputs.append(cluster_output)

        # Concatenate the augmented hidden states of the ESN cells

        esn_outputs_concatenated = self.concat_esn_layer(esn_outputs)

        output = self.concat_input_layer([inputs, esn_outputs_concatenated])

        return output

    # # # # # # Method for plotting the model

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the model.

        Args:
            input_shape (tuple): Input shape of the model.

        Returns:
            tuple: Output shape of the model.

        """
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2]
            + self.reservoir_amount * self.units_per_reservoir
            + 2 * self.overlap,
        )

    def build_graph(self):
        """Return a dummy model to use for plotting the model graph."""
        assert self.built, "Model must be built before calling build_graph()"
        dummy_input = keras.Input(shape=self.inputshape, name="Input")
        return keras.Model(
            inputs=[dummy_input], outputs=self.call(dummy_input)
        )

    # # # # # # Methods for ESN cells

    # Get the states of the ESN cells
    def get_esn_states(self):
        """Get the states of the ESN cells."""
        states = []
        for esn_layer in self.esn_layers:
            states.append(esn_layer.get_esn_state())
        return states

    # Set the states of the ESN cells
    def set_esn_states(self, states):
        """Set the states of the ESN cells.

        Args:
            states (list): List of states of the ESN cells.
        """
        for i, esn_layer in enumerate(self.esn_layers):
            esn_layer.set_esn_states(states[i])

    # Reset the states of the ESN cells
    def reset_esn_states(self):
        """Reset the states of the ESN cells."""
        for esn_layer in self.esn_layers:
            esn_layer.reset_esn_states()

    # # # # # # Pretraining methods

    def verify_esp(self, transient_data, times=1, verbose=1):
        """Verify the Echo State Property (ESP) of the model.

        Args:
            transient_data (tf.Tensor): Transient data of shape (batch_size, time_steps, input_dim).

            times (int): Number of times the transient data is used to verify the ESP.
        Returns:
            bool: True if the ESP is verified, False otherwise.
        """
        if not self.built:
            print("Model is not built yet, building model...")
            self.build(transient_data.shape)

        split_transient = self.input_split(transient_data)

        for i, esn_layer in enumerate(self.esn_layers):
            if verbose > 0:
                print(
                    f"Verifying ESP for reservoir {i+1}/{self.reservoir_amount}\n"
                )

            esp = esn_layer.verify_esp(
                split_transient[i], times, verbose=verbose
            )
            if not esp:
                return False

        return True

    def get_weights(self):
        """Get the weights of the model."""
        weights = []
        for esn_layer in self.esn_layers:
            weights.append(esn_layer.get_weights())
        return weights

    # # # # # # Methods for saving/loading the model (serializing)

    def get_config(self):
        """Get the configuration of the model."""
        config = super().get_config().copy()
        config.update(
            {
                "units_per_reservoir": self.units_per_reservoir,
                "reservoir_amount": self.reservoir_amount,
                "overlap": self.overlap,
                "leak_rate": self.leak_rate,
                "input_reservoir_init": self.input_reservoir_init,
                "input_bias_init": self.input_bias_init,
                "reservoir_kernel_init": self.reservoir_kernel_init,
                "esn_activation": self.esn_activation,
                "exponent": self.exponent,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Generic model with readout layer
@tf.keras.utils.register_keras_serializable(package="custom")
class ModelWithReadout(keras.Model):
    """Create a model with a readout layer.

    Args:
        model (keras.Model): The model to add the readout to.
        readout (keras.layers.Layer): The readout layer.

    Returns:
        keras.Model: The model with the readout layer.
    """

    def __init__(self, model, readout, **kwargs):
        """Initialize the model with the readout layer."""
        super().__init__(**kwargs)
        self.model = model
        self.readout = readout

    def call(self, inputs):
        """Call the model.

        Args:
            inputs (tf.Tensor): The input to the model.

        Returns:
            tf.Tensor: The output of the model.
        """
        output = self.model(inputs)
        return self.readout(output)

    def build_graph(self):
        """Generate dummy model for plotting.

        Returns:
            keras.Model: dummy model for plotting and summary
        """
        assert self.built, "Model must be built before calling build_graph()"

        model = self.model.build_graph()

        output = self.readout(model.output)

        return keras.Model(inputs=model.inputs, outputs=output)

    def get_config(self):
        """Get the config dictionary of the model for serialization."""
        config = super().get_config().copy()
        config.update(
            {
                "model": self.model,
                "readout": self.readout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# A model similar to ESN model, with an input-to-reservoir layer, but instead of using an EsnCell
# it uses a ReservoirCell in the recurrent layer
class ReservoirModel(keras.Model):
    """
    A simple ESN model.

    Args:
        units: Number of units in the internal state of the reservoir.

        leak_rate: Leaky rate of the reservoir

        input_reservoir_init: Initializer for the input to reservoir weights.
            Defaults to InputMatrix()

        input_bias_init: Initializer for the input bias.
            Defaults to RandomUniform()

        reservoir_function: The function to be used as reservoir in the recurrent layer. This is used in the ReservoirCell

        esn_activation: Activation function of the reservoir.
            Defaults to tanh.

        exponent: Exponent of the power function applied to the reservoir.
            Defaults to 2.

        seed: Seed of the model

        **kwargs: Keyword arguments for the keras.Model class

    Returns:
        A keras.Model object

    Example usage:

    >>> model = ReservoirModel(units=100,
                         leak_rate=0.5,
                         reservoir_function=lamda x: x**2,
                         reservoir_kernel_init='ErdosRenyi',
                         reservoir_bias_init='zeros',
                         esn_activation='tanh',
                         exponent=2,
                         seed=42)
    """

    def __init__(
        self,
        units,
        leak_rate,
        input_reservoir_init,
        input_bias_init,
        reservoir_function,
        esn_activation,
        exponent,
        seed,
        **kwargs,
    ):
        """Initialize the model."""
        super().__init__(**kwargs)
        self.units = units
        self.leak_rate = leak_rate
        self.input_reservoir_init = input_reservoir_init
        self.input_bias_init = input_bias_init
        self.reservoir_function = reservoir_function
        self.esn_activation = esn_activation
        self.exponent = exponent
        self.seed = seed

        self.input_to_reservoir = keras.layers.Dense(
            self.units,
            kernel_initializer=self.input_reservoir_init,
            bias_initializer=self.input_bias_init,
            use_bias=True,
            name="input_to_reservoir",
        )

        self.reservoir_layer = keras.layers.RNN(
            ReservoirCell(
                units=self.units,
                reservoir_function=self.reservoir_function,
                esn_activation=self.esn_activation,
                leak_rate=self.leak_rate,
                seed=self.seed,
            ),
            return_sequences=True,
            return_state=True,
            name="reservoir_layer",
        )

    def build(self, input_shape):
        """Build the model.

        Args:
            input_shape (tuple): Shape of the input data.
        """
        super().build(input_shape)

    def call(self, inputs):
        """Call the model.

        Args:
            inputs (tf.Tensor): Input data.

        Returns:
            tf.Tensor: Output of the model.
        """
        output = self.input_layer(inputs)
        output = self.input_to_reservoir(output)
        output = self.reservoir_layer(output)
        return output

    def build_graph(self):
        """Build the model graph.

        Returns:
            keras.Model: The model graph.
        """
        assert self.built, "Model must be built before calling build_graph()"

        input_layer = self.input_layer.build_graph()
        input_to_reservoir = self.input_to_reservoir.build_graph()
        reservoir_layer = self.reservoir_layer.build_graph()

        output = input_to_reservoir(input_layer.output)
        output = reservoir_layer(output)

        return keras.Model(inputs=input_layer.input, outputs=output)

    def get_config(self):
        """Get the config dictionary of the model for serialization."""
        config = super().get_config().copy()
        config.update(
            {
                "units": self.units,
                "leak_rate": self.leak_rate,
                "input_reservoir_init": self.input_reservoir_init,
                "input_bias_init": self.input_bias_init,
                "reservoir_function": self.reservoir_function,
                "esn_activation": self.esn_activation,
                "exponent": self.exponent,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


custom_models = {
    "ESN": ESN,
    "ParallelESN": ParallelESN,
    "ModelWithReadout": ModelWithReadout,
}

keras.utils.get_custom_objects().update(custom_models)
