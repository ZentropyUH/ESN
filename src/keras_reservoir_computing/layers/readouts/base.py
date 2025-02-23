from abc import ABC, abstractmethod

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="krc", name="ReadOut")
class ReadOut(keras.Layer, ABC):
    """
    Base class for readout layers in a reservoir.

    This class should be subclassed when implementing a new readout layer.
    The subclass should implement the `build`, `call` and `fit` methods.

    Parameters
    ----------
    units : int
        Number of outputs (a.k.a. the dimension of y).
    **kwargs : dict
        Additional keyword arguments passed to the base Layer class.

    Attributes
    ----------
    units : int
        Number of outputs.
    """

    def __init__(
        self, units: int, washout: int = 0, trainable: bool = False, **kwargs
    ) -> None:
        """
        Initialize the ReadOut layer.

        Parameters
        ----------
        units : int
            Number of outputs (a.k.a. the dimension of y).
        **kwargs : dict
            Additional keyword arguments for the Layer base class.
        """
        super().__init__(trainable=trainable, **kwargs)
        self.units = units
        self.washout = washout
        self._fitted = False

    @abstractmethod
    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Create the weights of the readout layer.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs.
        """
        pass

    @abstractmethod
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the readout layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, timesteps, features).

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, timesteps, units).
        """
        pass

    @abstractmethod
    def fit(X: tf.Tensor, Y: tf.Tensor) -> None:
        """
        Fit the readout layer to the data. This method can be called or not optionally.

        Parameters
        ----------
        X : tf.Tensor
            Input data of shape (batch_size, timesteps, features).
        Y : tf.Tensor
            Target data of shape (batch_size, timesteps, units).
        """
        pass

    def get_config(self) -> dict:
        """
        Allows serialization of the layer.

        Returns
        -------
        dict
            Configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {"units": self.units, "washout": self.washout, "trainable": self.trainable}
        )
        return config

    def get_params(self) -> dict:
        """
        Return a dict of key parameters.

        Returns
        -------
        dict
            Key parameters of the layer.
        """
        return {"units": self.units}
