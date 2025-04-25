from typing import Callable, Union

import tensorflow as tf

from keras_reservoir_computing.layers.reservoirs.layers.base import BaseReservoir
from keras_reservoir_computing.layers.reservoirs.cells import ESNCell


@tf.keras.utils.register_keras_serializable(package="krc", name="ESNReservoir")
class ESNReservoir(BaseReservoir):
    """
    An Echo State Network (ESN) reservoir layer implementation.

    This layer implements the reservoir component of an Echo State Network, which is a type of
    recurrent neural network where the internal weights (reservoir) remain fixed after
    initialization. The reservoir provides a rich, nonlinear transformation of the inputs
    through its recurrent connections.

    Parameters
    ----------
    units : int
        Number of units (neurons) in the reservoir.
    feedback_dim : int, optional
        Dimensionality of the feedback input. Default is 1.
    input_dim : int, optional
        Dimensionality of the external input. If 0, the reservoir only receives feedback.
        Default is 0.
    leak_rate : float, optional
        Leaking rate of the reservoir neurons. Must be between 0 and 1.
        A smaller value creates more memory in the reservoir. Default is 1.0.
    activation : str or callable, optional
        Activation function for the reservoir neurons. Default is "tanh".
    input_initializer : str or callable, optional
        Initializer for the input weights. Default is "glorot_uniform".
    feedback_initializer : str or callable, optional
        Initializer for the feedback weights. Default is "glorot_uniform".
    feedback_bias_initializer : str or callable, optional
        Initializer for the feedback bias. Default is "glorot_uniform".
    kernel_initializer : str or callable, optional
        Initializer for the reservoir's internal weights. Default is "glorot_uniform".
    dtype : str, optional
        Data type for the layer, by default "float32".
    **kwargs : dict
        Additional keyword arguments passed to the parent RNN layer.

    Notes
    -----
    - The reservoir can receive two types of inputs: external inputs and feedback inputs.
      Hence can be used in two modes:
        1. Feedback only: The reservoir receives only the feedback input.
        2. [Feedback, input]: The reservoir receives both feedback and external inputs. Pretty much like a Concatenate layer. If two inputs are provided, the feedback input must be the first one and both have the first two dimensions (batch_size, timesteps) equal.
    - The `feedback_dim` and `input_dim` parameters are crucial for differentiating
      between the two internally in the cell implementation.
    - The layer always outputs the full sequence of reservoir states.

    References
    ----------
    .. [1] Jaeger, H. (2001). The "echo state" approach to analysing and training
           recurrent neural networks. German National Research Center for Information
           Technology GMD Technical Report, 148(34), 13.
    """

    def __init__(
        self,
        units: int,
        feedback_dim: int = 1,
        input_dim: int = 0,
        leak_rate: float = 1.0,
        activation: Union[str, Callable] = "tanh",
        input_initializer: Union[str, Callable] = "zeros",
        feedback_initializer: Union[str, Callable] = "glorot_uniform",
        feedback_bias_initializer: Union[str, Callable] = "zeros",
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        dtype: str = "float32",
        **kwargs,
    ) -> None:

        cell = ESNCell(
            units=units,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            leak_rate=leak_rate,
            activation=activation,
            input_initializer=input_initializer,
            feedback_initializer=feedback_initializer,
            feedback_bias_initializer=feedback_bias_initializer,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
        )
        super().__init__(cell=cell, dtype=dtype, **kwargs)

    @property
    def units(self):
        return self.cell.units

    @property
    def feedback_dim(self):
        return self.cell.feedback_dim

    @property
    def input_dim(self):
        return self.cell.input_dim

    @property
    def leak_rate(self):
        return self.cell.leak_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def input_initializer(self):
        return self.cell.input_initializer

    @property
    def feedback_initializer(self):
        return self.cell.feedback_initializer

    @property
    def feedback_bias_initializer(self):
        return self.cell.feedback_bias_initializer

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer



    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "feedback_dim": self.feedback_dim,
            "input_dim": self.input_dim,
            "leak_rate": self.leak_rate,
            "activation": tf.keras.activations.serialize(self.activation),
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
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
