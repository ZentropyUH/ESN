from typing import Dict, Union

import keras
import tensorflow as tf

from .base import ReadOut


@keras.saving.register_keras_serializable(package="krc", name="RidgeSVDReadout")
class MoorePenroseReadout(ReadOut):
    """
    Moore-Penrose pseudoinverse readout layer for reservoir computing.

    Parameters
    ----------
    units : int
        Number of output units.
    alpha : float, optional
        Regularization strength, must be non-negative (default is 1.0).
    washout : int, optional
        Number of initial time steps to discard during training (default is 0).
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
        input_dtype = inputs.dtype
        inputs = tf.cast(inputs, tf.float64)
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        outputs = tf.cast(outputs, input_dtype)
        return outputs

    @tf.function()
    def fit(self, X: tf.Tensor, y: tf.Tensor) -> None:
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float64)
        else:
            X = tf.cast(X, tf.float64)

        if not isinstance(y, tf.Tensor):
            y = tf.convert_to_tensor(y, dtype=tf.float64)
        else:
            y = tf.cast(y, tf.float64)


        X = tf.reshape(X, (-1, X.shape[-1]))
        y = tf.reshape(y, (-1, y.shape[-1]))

        n_samples, n_features = X.shape
        if not self.built:
            self.build(X.shape)

        if y.shape[-1] != self.units:
            raise ValueError(
                f"Expected y to have shape (n_samples, {self.units}), "
                f"but got {y.shape} instead."
            )

        # Center the data
        X_mean = tf.reduce_mean(X, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Compute Moore-Penrose pseudoinverse with ridge regularization
        # identity = tf.eye(n_features, dtype=tf.float64)
        # xTx = tf.matmul(X_centered, X_centered, transpose_a=True) + self._alpha * identity
        # xTx_inv = tf.linalg.pinv(xTx)  # Moore-Penrose pseudoinverse
        # coef = tf.matmul(xTx_inv, tf.matmul(X_centered, y_centered, transpose_a=True))



        # Compute SVD of X_centered
        s, u, v = tf.linalg.svd(X_centered, full_matrices=False)

        # Compute the pseudoinverse of the singular values with regularization
        s_inv = tf.linalg.diag(1.0 / (s + self._alpha))

        # Compute pseudoinverse using SVD
        X_pinv = tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))

        # Compute coefficients
        coef = tf.matmul(X_pinv, y_centered)


        # Compute intercept
        intercept = y_mean - tf.matmul(X_mean, coef)
        intercept = tf.reshape(intercept, [self.units])

        # Assign values
        self.kernel.assign(coef)
        self.bias.assign(intercept)
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
        config.update({
            "units": self.units,
            "alpha": self._alpha,
            "trainable": self.trainable
        })
        return config
