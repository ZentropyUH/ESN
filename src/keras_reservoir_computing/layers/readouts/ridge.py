from typing import Dict, Tuple, Union, Optional

import tensorflow as tf

from keras_reservoir_computing.utils.tensorflow import tf_function

from .base import ReadOut


@tf.keras.utils.register_keras_serializable(package="krc", name="RidgeReadout")
class RidgeReadout(ReadOut):
    """
    A Keras-like, Conjugate Gradient-based Ridge Regression layer.

    This layer behaves much like a Dense layer, except that
    the actual weights are determined by a closed-form solution.

    Parameters
    ----------
    units : int
        Number of outputs (a.k.a. the dimension of y).
    alpha : float, optional
        L2 regularization strength. Must be non-negative. Default is 1.0.
    max_iter : int, optional
        Maximum number of iterations for the conjugate gradient solver. Default is 1000.
    tol : float, optional
        Tolerance for the solution of the conjugate gradient solver. Default is 1e-6.
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
        max_iter: Optional[int] = None,
        tol: float = 1e-6,
        trainable: bool = False,
        **kwargs,
    ):

        if alpha < 0:
            raise ValueError("Regularization strength `alpha` must be non-negative.")
        if max_iter is None:
            self._max_iter = max(1000, units * 10)
        elif max_iter <= 0:
            raise ValueError("`max_iter` must be positive.")
        else:
            self._max_iter = max_iter

        if tol <= 0:
            raise ValueError("`tol` must be positive.")
        self._alpha = alpha
        self._tol = tol

        # Will be created in build() with known shapes.
        self.kernel = None  # shape: (input_dim, units)
        self.bias = None  # shape: (units,)

        self._fitted = False

        super().__init__(units=units, trainable=trainable, **kwargs)

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

        # Make sure inputs are same dtype as kernel/bias
        # Only cast if needed, otherwise keep original dtype for better performance
        if inputs.dtype != self.kernel.dtype:
            inputs = tf.cast(inputs, self.kernel.dtype)
        outputs = tf.matmul(inputs, self.kernel) + self.bias

        # Only cast back if input was a different dtype
        if input_dtype != self.kernel.dtype:
            outputs = tf.cast(outputs, input_dtype)
        return outputs

    def _fit(self, X: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        alpha = tf.constant(self._alpha, dtype=tf.float64)
        units = tf.constant(self.units, dtype=tf.int32)
        max_iter = tf.constant(self._max_iter, dtype=tf.int32)
        tol = tf.constant(self._tol, dtype=tf.float64)
        return self.__class__._solve_ridge_cg(X, y, alpha, units, max_iter, tol)


    @staticmethod
    @tf_function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),  # X
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),  # y
            tf.TensorSpec(shape=(), dtype=tf.float64),            # alpha
            tf.TensorSpec(shape=(), dtype=tf.int32),              # units
            tf.TensorSpec(shape=(), dtype=tf.int32),              # max_iter
            tf.TensorSpec(shape=(), dtype=tf.float64),            # tol
            ]
        )
    def _solve_ridge_cg(
        X: tf.Tensor, 
        y: tf.Tensor, 
        alpha: float, 
        units: int,
        max_iter: int,
        tol: float,
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Solve Ridge Regression using Conjugate Gradient for multiple outputs.

        This method solves the regularized least squares problem:

            (X.T @ X + alpha * I) @ W = X.T @ Y

        where:
        - X is the input matrix of shape (n_samples, n_features),
        - Y is the target matrix of shape (n_samples, units),
        - W is the solution matrix of shape (n_features, units).

        Each output dimension is solved independently using a vectorized
        implementation of the Conjugate Gradient (CG) method. Scalar updates
        (alpha, beta) are computed per output column using broadcasting, allowing
        the solver to run without explicit Python loops and remain compatible
        with TensorFlow's graph execution and GPU acceleration.

        This function also centers the data and returns the corresponding bias term.

        Notes
        -----
        - Works with float64 precision only (for numerical stability).
        - Fully compatible with @tf.function and Autograph tracing.
        - Efficient for high-dimensional regression with many outputs.
        - This is a mathematically correct but non-standard batched CG trick;
        do not refactor it without understanding how the scalar updates are applied.

        Parameters
        ----------
        X : tf.Tensor
            Input tensor of shape (n_samples, n_features).
        y : tf.Tensor
            Target tensor of shape (n_samples, units).
        alpha : float
            L2 regularization strength. Must be non-negative.
        units : int
            Number of output units (columns in `y`).

        Returns
        -------
        coefs : tf.Tensor
            Solution weight matrix of shape (n_features, units).
        intercept : tf.Tensor
            Bias term of shape (units,).

        Raises
        ------
        ValueError
            If `alpha` is negative.
        """
        X_mean = tf.reduce_mean(X, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
        Xc = X - X_mean
        yc = y - y_mean

        def matvec(w):
            # w: (n_features, n_units)
            Xw = tf.matmul(Xc, w)  # (n_samples, n_units)
            XtXw = tf.matmul(Xc, Xw, transpose_a=True)  # (n_features, n_units)
            return XtXw + alpha * w

        def cg(A, B, max_iter=max_iter, tol=tol):
            X = tf.zeros_like(B)
            R = B - A(X)
            P = R
            Rs_old = tf.reduce_sum(R * R, axis=0, keepdims=True)

            for _ in tf.range(max_iter):
                AP = A(P)
                alpha_cg = Rs_old / tf.reduce_sum(P * AP, axis=0, keepdims=True)
                X = X + P * alpha_cg
                R = R - AP * alpha_cg
                Rs_new = tf.reduce_sum(R * R, axis=0, keepdims=True)
                if tf.reduce_all(Rs_new < tol**2):
                    break
                beta = Rs_new / Rs_old
                P = R + P * beta
                Rs_old = Rs_new
            return X

        rhs = tf.matmul(Xc, yc, transpose_a=True)  # (n_features, units)
        coefs = cg(matvec, rhs)

        intercept = tf.reshape(y_mean - tf.matmul(X_mean, coefs), [units])
        return coefs, intercept

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
        config.update(
            {"units": self.units, "alpha": self._alpha, "trainable": self.trainable, "max_iter": self._max_iter, "tol": self._tol}
        )
        return config
