from typing import Callable, List, Optional, Union
from .base import BaseCell
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="MyLayers", name="ESNCell")
class ESNCell(BaseCell):
    """
    A custom RNN cell that splits the incoming features into:
      - feedback portion (first feedback_dim)
      - input portion (next input_dim)
    Then applies a state update.

    states_{t} = activation(
       feedback_t * W_fb + input_t * W_in + states_{t-1} * W_kernel
    )
    """

    def __init__(
        self,
        units: int,
        feedback_dim: int = 1,
        input_dim: int = 0,
        leak_rate: float = 1.0,
        # These are not handed by parent class
        activation: Optional[Union[str, Callable]] = "tanh",
        noise_level: float = 0.0,
        input_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        feedback_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        feedback_bias_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        kernel_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        **kwargs
    ) -> None:

        super().__init__(
            units=units,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            leak_rate=leak_rate,
            **kwargs
        )

        self.activation = tf.keras.activations.get(activation)
        self.noise_level = noise_level
        self.input_initializer = tf.keras.initializers.get(input_initializer)
        self.feedback_initializer = tf.keras.initializers.get(feedback_initializer)
        self.feedback_bias_initializer = tf.keras.initializers.get(
            feedback_bias_initializer
        )
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape: tf.TensorShape):
        """
        Keras will pass input_shape of the form (batch_size, timesteps, total_features)
        for RNN cells. total_features = feedback_dim + input_dim in your usage.
        """

        # We'll create three sets of weights:
        #   W_fb: (feedback_dim, units)
        #   W_in: (input_dim, units)
        #   W_kernel: (units, units)  -- the "recurrent" part
        self.W_fb = self.add_weight(
            shape=(self.feedback_dim, self.units),
            initializer=self.feedback_initializer,
            name="W_fb",
            trainable=False,
        )

        if self.input_dim > 0:
            self.W_in = self.add_weight(
                shape=(self.input_dim, self.units),
                initializer=self.input_initializer,
                name="W_in",
                trainable=False,
            )

        self.W_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            name="W_kernel",
            trainable=False,
        )

        self.b_fb = self.add_weight(
            shape=(self.units,),
            initializer=self.feedback_bias_initializer,
            name="b_fb",
            trainable=False,
        )

        # Spurious call to super().build() to make Keras happy. Also spurious use of input_shape. TODO: see if this is a problem.
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, states: List[tf.Tensor]):
        """
        inputs:  shape (batch_size, total_features)
                 total_features = feedback_dim + input_dim
        states:  a list of one tensor [previous_state], shape (batch_size, units)
        """
        prev_state = states[0]

        # Split out the feedback portion vs. external input portion
        feedback_part = inputs[:, : self.feedback_dim]

        input_part = inputs[:, self.feedback_dim :]  # remainder

        # Compute new state

        next_state = tf.matmul(feedback_part, self.W_fb) + self.b_fb

        if self.input_dim > 0:
            next_state += tf.matmul(input_part, self.W_in)

        next_state += tf.matmul(prev_state, self.W_kernel)

        # Apply activation
        next_state = self.activation(next_state)

        # Add noise TODO: This need modification, training=True only when model.fit is called, we want it to be True when model.predict is called controlled by hand.
        next_state += tf.random.normal(
            tf.shape(next_state), mean=0.0, stddev=self.noise_level
        )

        # leaky integration
        next_state = (1 - self.leak_rate) * prev_state + self.leak_rate * next_state

        # Return (output, new_state). For a basic RNN, the "output" is usually
        # the same as new_state.
        return next_state, [next_state]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "noise_level": self.noise_level,
                "input_initializer": tf.keras.initializers.serialize(
                    self.input_initializer
                ),
                "feedback_initializer": tf.keras.initializers.serialize(
                    self.feedback_initializer
                ),
                "feedback_bias_initializer": tf.keras.initializers.serialize(
                    self.feedback_bias_initializer
                ),
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
