from typing import Dict, Union

import tensorflow as tf

from .base import ReadOut


class RidgeSVDReadout(ReadOut):
    """
    A Keras-like, SVD-based Ridge Regression layer.

    This layer behaves much like a Dense layer, except that
    the actual weights are determined by a closed-form solution.

    Parameters
    ----------
    units : int
        Number of outputs (a.k.a. the dimension of y).
    alpha : float, optional
        L2 regularization strength. Must be non-negative. Default is 1.0.
    trainable : bool, optional
        Whether to allow gradient-based updates on the weights
        after they are fit with the closed-form solver. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to the base Layer class.

    Attributes
    ----------
    units : int
        Number of outputs.
    alpha_value : float
        Regularization strength.
    kernel : tf.Tensor or None
        Weight matrix of shape (input_dim, units). Initialized in `build()`.
    bias : tf.Tensor or None
        Bias vector of shape (units,). Initialized in `build()`.
    _fitted : bool
        Whether the layer has been fitted.

    Methods
    -------
    build(input_shape)
        Create the kernel and bias.
    call(inputs)
        Forward pass: outputs = inputs @ kernel + bias.
    fit(X, y)
        Compute the closed-form Ridge solution via SVD.
    alpha()
        Returns the current regularization parameter.
    alpha(value)
        Update alpha and invalidate the current fit.
    fitted_()
        Returns True if fit() has been called successfully at least once.
    get_params()
        Return a dict of key parameters (alpha, kernel, bias).
    get_config()
        Allows serialization of the layer.

    Notes
    -----
    - This layer is designed to be used as a readout layer in a reservoir. It can however be integrated in any Keras model.
    - The layer is not trainable by default, but can be made trainable by setting `trainable=True`. In which case, the weights will be updated by gradient descent, but can still be re-fitted with the closed-form solution. This can be useful for fine-tuning, using the closed form solution as a warm start.
    """

    def __init__(
        self,
        units: int,
        alpha: float = 1.0,
        washout: int = 0,
        trainable: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        units : int
            Number of outputs (a.k.a. the dimension of y).
        alpha : float
            L2 regularization strength. Must be non-negative.
        trainable : bool
            Whether to allow gradient-based updates on the weights
            after they are fit with the closed-form solver.
        washout : int
            Number of time steps to ignore during training.
        """
        if alpha < 0:
            raise ValueError("Regularization strength `alpha` must be non-negative.")

        self._alpha = alpha

        # Will be created in build() with known shapes.
        self.kernel = None  # shape: (input_dim, units)
        self.bias = None  # shape: (units,)
        
        self._fitted = False

        super().__init__(units=units, washout=washout, trainable=trainable, **kwargs)

    def build(self, input_shape) -> None:
        """
        Create the kernel and bias. We do this once we know input_dim.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
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

        super().build(input_shape)  # Keras housekeeping

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass: outputs = inputs @ kernel + bias
        This is identical to a standard Dense layer's operation.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        input_dtype = inputs.dtype

        # Make sure inputs are float64
        inputs = tf.cast(inputs, tf.float64)
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        outputs = tf.cast(outputs, input_dtype)
        return outputs

    @tf.function()
    def fit(self, X: tf.Tensor, y: tf.Tensor) -> "RidgeSVDReadout":
        """
        Compute the closed-form Ridge solution via SVD:

            coef = V * diag(1 / (s + alpha)) * U^T * y

        Then assign it to `kernel`, and compute intercept to assign to `bias`.

        Parameters
        ----------
        X : tf.Tensor
            Input tensor of shape (n_samples, n_features).
        y : tf.Tensor
            Target tensor of shape (n_samples, units).

        Returns
        -------
        TFRidgeLayer
            The fitted layer.
        """
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float64)
        else:
            X = tf.cast(X, tf.float64)

        if not isinstance(y, tf.Tensor):
            y = tf.convert_to_tensor(y, dtype=tf.float64)
        else:
            y = tf.cast(y, tf.float64)

        # Remove washout
        X = X[self.washout :]
        y = y[self.washout :]

        # Ensure shape
        X = tf.reshape(X, (-1, X.shape[-1]))  # (n_samples, n_features)
        y = tf.reshape(y, (-1, y.shape[-1] if len(y.shape) > 1 else 1))

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

        # Center the data
        X_mean = tf.reduce_mean(X, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # SVD of centered X
        s, U, V = tf.linalg.svd(X_centered, full_matrices=False)

        # Regularize singular values
        eps = tf.keras.backend.epsilon()
        threshold = eps * tf.reduce_max(s)

        s_inv = tf.math.reciprocal_no_nan(s + self._alpha)
        s_inv = tf.where(s > threshold, s_inv, 0.0)

        s_inv = tf.reshape(s_inv, (-1, 1))

        # U^T y_centered
        UTy = tf.matmul(U, y_centered, transpose_a=True)

        # Coefficients: shape (n_features, units)
        coef = tf.matmul(V, s_inv * UTy, transpose_b=False)

        # Intercept: shape (1, units) -> flatten to shape (units,)
        intercept = y_mean - tf.matmul(tf.reshape(X_mean, [1, -1]), coef)
        intercept = tf.reshape(intercept, [self.units])  # Explicit reshape

        # Assign to kernel/bias
        self.kernel.assign(coef)
        self.bias.assign(intercept)

        self._fitted = True

    @property
    def alpha(self) -> float:
        """Returns the current regularization parameter."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """
        Update alpha and invalidate the current fit.

        Parameters
        ----------
        value : float
            New regularization strength. Must be non-negative.
        """
        if value < 0:
            raise ValueError("Regularization strength must be non-negative.")
        self._alpha = float(value)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """Returns True if fit() has been called successfully at least once."""
        return self._fitted

    def get_params(self) -> Dict[str, Union[float, tf.Tensor, None]]:
        """
        Return a dict of key parameters (alpha, kernel, bias).

        Returns
        -------
        Dict[str, Union[float, tf.Tensor, None]]
            Dictionary of key parameters.
        """
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
            "washout": self.washout,
            "trainable": self.train
        })
        return config