from typing import Callable, List, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.cells.base import BaseCell


@tf.keras.utils.register_keras_serializable(package="krc", name="ESNCell")
class ESNCell(BaseCell):
    """
    Echo State Network (ESN) cell implementation for reservoir computing.

    This cell implements the ESN architecture where input features are split into
    feedback and input portions, then applies a state update according to:
    states_{t} = activation(feedback_t * W_fb + input_t * W_in + states_{t-1} * W_kernel)

    Parameters
    ----------
    units : int
        Number of units in the reservoir cell.
    feedback_dim : int, optional
        Dimensionality of the feedback input, by default 1.
    input_dim : int, optional
        Dimensionality of the external input, by default 0.
    leak_rate : float, optional
        Leak rate for the leaky integration, by default 1.0.
    activation : Optional[Union[str, Callable]], optional
        Activation function to use, by default "tanh".
    input_initializer : Optional[Union[str, Callable]], optional
        Initializer for the input weights, by default "zeros".
    feedback_initializer : Optional[Union[str, Callable]], optional
        Initializer for the feedback weights, by default "glorot_uniform".
    feedback_bias_initializer : Optional[Union[str, Callable]], optional
        Initializer for the feedback bias, by default "zeros".
    kernel_initializer : Optional[Union[str, Callable]], optional
        Initializer for the recurrent kernel, by default "glorot_uniform".
    """

    def __init__(
        self,
        units: int,
        feedback_dim: int = 1,
        input_dim: int = 0,
        leak_rate: float = 1.0,
        # Additional parameters handled by this class (not by parent class)
        activation: Union[str, Callable] = "tanh",
        input_initializer: Union[str, Callable] = "zeros",
        feedback_initializer: Union[str, Callable] = "glorot_uniform",
        feedback_bias_initializer: Union[str, Callable] = "zeros",
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        dtype: str = "float32",
        **kwargs,
    ) -> None:

        super().__init__(
            units=units,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            leak_rate=leak_rate,
            dtype=dtype,
            **kwargs,
        )

        self.activation = tf.keras.activations.get(activation)
        self.input_initializer = tf.keras.initializers.get(input_initializer)
        self.feedback_initializer = tf.keras.initializers.get(feedback_initializer)
        self.feedback_bias_initializer = tf.keras.initializers.get(feedback_bias_initializer)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape: tuple) -> None:
        """
        Build the cell's weights.

        Keras will pass input_shape of the form (batch_size, timesteps, total_features)
        for RNN cells. total_features = feedback_dim + input_dim in your usage.

        Parameters
        ----------
        input_shape : tuple
            The input shape provided by Keras.

        Raises
        ------
        ValueError
            If input shape doesn't match expected feedback_dim + input_dim.
        """

        feedback_and_input_features = input_shape[-1]

        # Check if the input shape has the correct number of features
        if feedback_and_input_features != self.feedback_dim + self.input_dim:
            raise ValueError(
                f"Input shape {input_shape} has {feedback_and_input_features} features, expected {self.feedback_dim + self.input_dim}"
            )

        # We'll create four sets of weights:
        #   W_fb: (feedback_dim, units)
        #   b_fb: (units,)
        #   W_in: (input_dim, units)
        #   W_kernel: (units, units)  -- the "recurrent" part
        self.W_fb = self.add_weight(
            shape=(self.feedback_dim, self.units),
            initializer=self.feedback_initializer,
            name="W_fb",
            trainable=False,
            dtype=self.dtype,
        )

        if self.input_dim > 0:
            self.W_in = self.add_weight(
                shape=(self.input_dim, self.units),
                initializer=self.input_initializer,
                name="W_in",
                trainable=False,
                dtype=self.dtype,
            )

        self.W_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            name="W_kernel",
            trainable=False,
            dtype=self.dtype,
        )

        self.b_fb = self.add_weight(
            shape=(self.units,),
            initializer=self.feedback_bias_initializer,
            name="b_fb",
            trainable=False,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, states: List[tf.Tensor]) -> tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Process one step of the cell.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, total_features)
            total_features = feedback_dim + input_dim
        states : List[tf.Tensor]
            A list of one tensor [previous_state], shape (batch_size, units)

        Returns
        -------
        tuple[tf.Tensor, List[tf.Tensor]]
            A tuple (next_state, [next_state]) containing the output and new states
        """
        # Get previous state x_{t}
        prev_state = states[0]

        # Split out the feedback portion vs. external input portion
        feedback_part = inputs[:, : self.feedback_dim]

        input_part = inputs[:, self.feedback_dim :]  # remainder

        # Compute new state y_{t} * W_{fb} + b_{fb}
        next_state = tf.matmul(feedback_part, self.W_fb) + self.b_fb

        if self.input_dim > 0:
            # Compute input part
            # u_{t} * W_{in} + y_{t} * W_{fb} + b_{fb}
            next_state += tf.matmul(input_part, self.W_in)

        # Compute recurrent part
        # x_{t} * W_{kernel} + u_{t} * W_{in} + y_{t} * W_{fb} + b_{fb}
        next_state += tf.matmul(prev_state, self.W_kernel)

        # Apply activation
        # f(x_{t} * W_{kernel} + u_{t} * W_{in} + y_{t} * W_{fb} + b_{fb})
        next_state = self.activation(next_state)

        # leaky integration
        # x_{t+1} = (1 - \alpha) * x_{t} + \alpha * f(x_{t} * W_{kernel} + u_{t} * W_{in} + y_{t} * W_{fb} + b_{fb})
        next_state = (1 - self.leak_rate) * prev_state + self.leak_rate * next_state

        # Return (output, new_state). For a basic RNN, the "output" is usually
        # the same as new_state.
        return next_state, [next_state]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "input_initializer": tf.keras.initializers.serialize(self.input_initializer),
                "feedback_initializer": tf.keras.initializers.serialize(self.feedback_initializer),
                "feedback_bias_initializer": tf.keras.initializers.serialize(
                    self.feedback_bias_initializer
                ),
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config
