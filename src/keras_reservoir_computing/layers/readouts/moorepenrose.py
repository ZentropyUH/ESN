from typing import Dict, Union

import tensorflow as tf

from keras_reservoir_computing.utils.tensorflow import tf_function

from .base import ReadOut


@tf.keras.utils.register_keras_serializable(package="krc", name="MoorePenroseReadout")
class MoorePenroseReadout(ReadOut):
    """
    Moore-Penrose pseudoinverse readout layer for reservoir computing.

    Parameters
    ----------
    units : int
        Number of output units.
    alpha : float, optional
        Regularization strength, must be non-negative (default is 1.0).
    trainable : bool, optional
        Whether the layer's weights are trainable (default is False).
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    alpha : float
        Regularization strength.
    kernel : tf.Tensor or None
        Weight matrix of the layer.
    bias : tf.Tensor or None
        Bias vector of the layer.
    fitted : bool
        Whether the layer has been fitted.

    Methods
    -------
    build(input_shape)
        Creates the layer's weights.
    call(inputs)
        Computes the output of the layer.
    fit(X, y)
        Fits the layer to the given data using the Moore-Penrose pseudoinverse.
    get_params()
        Returns the parameters of the layer.
    get_config()
        Returns the configuration of the layer.
    """

    def __init__(
        self,
        units: int,
        alpha: float = 1.0,
        trainable: bool = False,
        **kwargs,
    ):
        if alpha < 0:
            raise ValueError("Regularization strength `alpha` must be non-negative.")

        self._alpha = alpha
        self.kernel = None
        self.bias = None

        self._fitted = False

        super().__init__(units=units, trainable=trainable, **kwargs)

    def build(self, input_shape) -> None:
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=self.trainable,
            dtype=tf.float64,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=self.trainable,
            dtype=tf.float64,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Convert to the same precision as the weights for consistent calculations
        if inputs.dtype != self.kernel.dtype:
            inputs = tf.cast(inputs, self.kernel.dtype)
        
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        outputs = tf.cast(outputs, inputs.dtype)
        return outputs

    @tf_function
    def _fit(self, X: tf.Tensor, y: tf.Tensor) -> None:
        """
        Fit the readout layer to the data using the Moore-Penrose pseudoinverse.

        Parameters
        ----------
        X : tf.Tensor
            Input data of shape (n_samples, n_features).
        y : tf.Tensor
            Target data of shape (n_samples, n_outputs).

        Returns
        -------
        None

        Notes
        -----
        - The Moore-Penrose pseudoinverse is computed using the singular value decomposition (SVD) of the input data.
        - A threshold of 1e-15 is used for numerical stability to filter out near-zero singular values.
        """
        # Ensure data is in correct dtype
        X = tf.cast(X, self.kernel.dtype)
        y = tf.cast(y, self.kernel.dtype)
        
        # Center the data
        X_mean = tf.reduce_mean(X, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Compute SVD of X_centered
        s, U, V = tf.linalg.svd(X_centered, full_matrices=False)

        # Threshold small singular values for numerical stability
        # This value is chosen to filter out very small values that might cause numerical instability
        threshold = 1e-15

        # More stable ridge regression formula
        s_inv = s / (s**2 + self._alpha)
        s_inv = tf.where(s > threshold, s_inv, 0.0)
        s_inv = tf.reshape(s_inv, (-1, 1))

        # U^T y_centered
        UTy = tf.matmul(U, y_centered, transpose_a=True)

        # Apply pseudoinverse transformation
        d_UT_y = tf.broadcast_to(s_inv, tf.shape(UTy)) * UTy

        # Compute coefficients
        coef = tf.matmul(V, d_UT_y)

        # Compute intercept
        intercept = y_mean - tf.matmul(X_mean, coef)
        intercept = tf.reshape(intercept, [self.units])

        # Assign values
        self.kernel.assign(coef)
        self.bias.assign(intercept)
        
        # Mark as fitted
        self._fitted = True

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if value < 0:
            raise ValueError("Regularization strength must be non-negative.")
        self._alpha = float(value)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> Dict[str, Union[float, tf.Tensor, None]]:
        return {"alpha": self._alpha, "kernel": self.kernel, "bias": self.bias}

    def get_config(self) -> dict:
        """
        Allows serialization of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {"units": self.units, "alpha": self._alpha, "trainable": self.trainable}
        )
        return config
