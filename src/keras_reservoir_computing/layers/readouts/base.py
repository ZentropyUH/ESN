from abc import ABC, abstractmethod

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="ReadOut")
class ReadOut(tf.keras.Layer, ABC):
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
        self, units: int, trainable: bool = False, **kwargs
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

    def fit(self, X: tf.Tensor, y: tf.Tensor) -> None:
        """
        Fit the readout layer to the data. This method can be called or not optionally.

        Parameters
        ----------
        X : tf.Tensor
            Input data of shape (batch_size, timesteps, features).
        Y : tf.Tensor
            Target data of shape (batch_size, timesteps, units).
        """
        X = tf.cast(X, dtype=tf.float64)
        y = tf.cast(y, dtype=tf.float64)

        X = tf.reshape(X, (-1, X.shape[-1]))  # (n_samples, n_features)
        y = tf.reshape(y, (-1, y.shape[-1]))

        n_samples, n_features = X.shape
        # Build layer if not built yet
        if not self.built:
            self.build(X.shape)

        # Check that y matches self.units
        if y.shape[-1] != self.units:
            raise ValueError(
                f"Expected y to have shape (n_samples, {self.units}), "
                f"but got {y.shape} instead."
            )

        self._fit(X, y)

        self._fitted = True

    @abstractmethod
    def _fit(self, X: tf.Tensor, y: tf.Tensor) -> None:
        """
        Fit the readout layer to the data.

        This is a private method that is called by `fit()`. Must end assigning the weights directly to the kernel and bias attributes.

        Parameters
        ----------
        X : tf.Tensor
            Input data of shape (n_samples, n_features).
        y : tf.Tensor
            Target data of shape (n_samples, units).
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
            {"units": self.units, "trainable": self.trainable}
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
