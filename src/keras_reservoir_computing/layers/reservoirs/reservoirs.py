from typing import Callable, Optional, Union

import keras

from .base import BaseReservoir
from .cells import ESNCell


@keras.saving.register_keras_serializable(package="krc", name="ESNReservoir")
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
    noise_level : float, optional
        Standard deviation of the Gaussian noise added to the state updates.
        Helps improve robustness. Default is 0.0.
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
        noise_level: float = 0.0,
        activation: Optional[Union[str, Callable]] = "tanh",
        input_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        feedback_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        feedback_bias_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        kernel_initializer: Optional[Union[str, Callable]] = "glorot_uniform",
        **kwargs,
    ) -> None:
        """
        Initialize the ESNReservoir.

        Parameters
        ----------
        units : int
            Number of units in the reservoir.
        feedback_dim : int, optional
            Dimensionality of the feedback input, by default 1.
        input_dim : int, optional
            Dimensionality of the input, by default 0.
        activation : Optional[Union[str, Callable]], optional
            Activation function to use, by default "tanh".
        **kwargs : dict
            Additional keyword arguments for the RNN base class.
        """

        self.units = units
        self.feedback_dim = feedback_dim
        self.input_dim = input_dim
        self.leak_rate = leak_rate
        self.noise_level = noise_level
        self.activation = keras.activations.get(activation)
        self.input_initializer = keras.initializers.get(input_initializer)
        self.feedback_initializer = keras.initializers.get(feedback_initializer)
        self.feedback_bias_initializer = keras.initializers.get(
            feedback_bias_initializer
        )
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        cell = ESNCell(
            units=units,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            leak_rate=leak_rate,
            noise_level=noise_level,
            activation=activation,
            input_initializer=input_initializer,
            feedback_initializer=feedback_initializer,
            feedback_bias_initializer=feedback_bias_initializer,
            kernel_initializer=kernel_initializer,
        )
        super().__init__(cell, **kwargs)

    def build(self, input_shape):
        """
        Keras will pass the 'input_shape' that this RNN layer sees at build-time.
        It might be:
         - a single shape (batch_size, timesteps, features)
         - a list of two shapes [ (batch_size, timesteps, fb_feats),
                                  (batch_size, timesteps, in_feats) ]
           or something partially defined (None for batch or timesteps).

        We must figure out the "combined" shape that our cell expects:
            (batch_size, timesteps, feedback_dim + input_dim)
        Then call super().build(...) with that shape so it can build the cell.
        """

        if all(isinstance(shape, (list, tuple)) for shape in input_shape):
            shape_fb, shape_in = map(tuple, input_shape)
            # shape_fb = (batch_size, timesteps, fb_feats)
            # shape_in = (batch_size, timesteps, in_feats)
            # Some or all dims might be None at this stage.

            # If 'input_dim' is supposed to be 0 or if the input is genuinely None,
            # we won't know in advance. Typically you want shape_in[-1] == self.input_dim,
            # shape_fb[-1] == self.feedback_dim.
            # We'll define an aggregated 'features' dimension for the cell.

            fb_feats = shape_fb[-1]

            if fb_feats != self.feedback_dim:
                raise ValueError(
                    f"Feedback sequence has {fb_feats} features, expected {self.feedback_dim}"
                )

            in_feats = shape_in[-1] if shape_in is not None else self.input_dim

            if in_feats is not None and in_feats != self.input_dim:
                raise ValueError(
                    f"Input sequence has {in_feats} features, expected {self.input_dim}"
                )

            combined_features = fb_feats + in_feats
            # Now define a synthetic shape to pass to super().build(...).
            # We only really need (batch_size, timesteps, combined_features).
            # The batch_size or timesteps might be None, but that’s fine.
            shape_total = (shape_fb[0], shape_fb[1], combined_features)

        else:
            # Single input shape. We assume it’s the feedback sequence only.
            shape_total = tuple(input_shape)
            if shape_total[-1] != self.feedback_dim:
                raise ValueError(
                    f"Feedback sequence has {shape_total[-1]} features, expected {self.feedback_dim}"
                )

        # Now let the RNN (and thus the cell) build weights for that total dimension.
        super().build(shape_total)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the reservoir.
        Since the RNN always returns the full sequence, the output shape is:
            (batch_size, timesteps, units)
        """
        if all(isinstance(shape, (list, tuple)) for shape in input_shape):
            feedback_shape = input_shape[0]  # feedback sequence shape
            batch_size, timesteps = feedback_shape[:2]
        else:
            # Single input case (feedback only)
            batch_size, timesteps = input_shape[:2]

        return (batch_size, timesteps, self.units)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "feedback_dim": self.feedback_dim,
            "input_dim": self.input_dim,
            "leak_rate": self.leak_rate,
            "noise_level": self.noise_level,
            "activation": keras.activations.serialize(self.activation),
            "input_initializer": keras.initializers.serialize(self.input_initializer),
            "feedback_initializer": keras.initializers.serialize(
                self.feedback_initializer
            ),
            "feedback_bias_initializer": keras.initializers.serialize(
                self.feedback_bias_initializer
            ),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        }
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
