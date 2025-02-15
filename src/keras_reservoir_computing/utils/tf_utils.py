import tensorflow as tf
from typing import Optional, Union, Dict


def create_tf_rng(
    seed: Optional[Union[int, tf.random.Generator]] = None
) -> tf.random.Generator:
    """
    Create and return a TensorFlow random number generator (RNG).

    Parameters
    ----------
    seed : int, tf.random.Generator, or None, optional
        - If an integer, a new deterministic generator is created using that seed.
        - If a tf.random.Generator, it is returned as is.
        - If None, a new generator is created from a non-deterministic state.

    Returns
    -------
    tf.random.Generator
        A TensorFlow random number generator instance.

    Raises
    ------
    TypeError
        If `seed` is neither an integer, a `tf.random.Generator`, nor None.

    Examples
    --------
    >>> # Creating a deterministic generator:
    >>> rng = create_tf_rng(seed=42)
    >>> sample = rng.normal(shape=(2, 2))

    >>> # Creating a non-deterministic generator:
    >>> rng2 = create_tf_rng()
    """
    # Decide how to create or return the generator
    if isinstance(seed, int):
        rg = tf.random.Generator.from_seed(seed)
    elif isinstance(seed, tf.random.Generator):
        rg = seed
    elif seed is None:
        rg = tf.random.Generator.from_non_deterministic_state()
    else:
        raise TypeError("`seed` must be an integer, tf.random.Generator, or None.")

    return rg


class TF_Ridge:
    """
    A simple Ridge Regression model implemented in TensorFlow using SVD.

    The ridge regression solution is computed with:
    W = (X^T X + alpha I)^{-1} X^T y
    but is actually solved via SVD for numerical stability:

    W = V * (S^{-1} * (U^T y)), with S^{-1} adjusted for the alpha regularization.

    Parameters
    ----------
    alpha : float
        Non-negative regularization strength.

    Attributes
    ----------
    alpha : float
        Regularization strength (L2 penalty). If changed, the model must be refitted.
    built : bool
        Indicates whether the model has been fitted at least once.
    coef_ : tf.Tensor or None
        Coefficients of the fitted model (shape: (features, outputs)).
    intercept_ : tf.Tensor or None
        Intercept of the fitted model (shape: (outputs,)).
    n_features_in : int or None
        Number of features in the training data.
    W : tf.Tensor or None
        Concatenated coefficients and intercept, used for predictions
        (shape: (features+1, outputs)).

    Methods
    -------
    fit(X, y)
        Fit the Ridge regression model on the data.
    predict(X)
        Generate predictions using the fitted model.
    get_params()
        Return a dictionary of model parameters, including `alpha`, `coef_`, `intercept_`.

    Notes
    -----
    - All computations are done with dtype float64 to reduce floating-point errors.
    - Centering is performed to ensure an unbiased estimate of the intercept.
    - If any alpha > 0, it shrinks singular values of X to mitigate overfitting.
    """

    def __init__(self, alpha: float) -> None:
        """
        Initialize the TF_Ridge model with a given regularization strength (alpha).

        Parameters
        ----------
        alpha : float
            Non-negative regularization strength.
        """
        if alpha < 0:
            raise ValueError("Regularization strength `alpha` must be non-negative.")
        if not isinstance(alpha, (int, float)):
            raise TypeError("`alpha` must be an integer or float.")

        self._alpha = float(alpha)
        self._built = False
        self._coef = None
        self._intercept = None
        self._W = None
        self._n_features_in = None

    def fit(self, X: tf.Tensor, y: tf.Tensor) -> "TF_Ridge":
        """
        Fit the Ridge regression model using SVD-based solution.

        The formula used is:

        W = V * diag(1 / (s + alpha)) * U^T * y

        where X = U * diag(s) * V^T is the SVD of X,
        and s are singular values.

        Parameters
        ----------
        X : tf.Tensor
            Input data of shape (n_samples, n_features).
        y : tf.Tensor
            Target data of shape (n_samples,) or (n_samples, n_targets).

        Returns
        -------
        TF_Ridge
            The fitted model (self).

        Raises
        ------
        TypeError
            If X or y is not a tf.Tensor.
        ValueError
            If alpha < 0 or model dimension doesn't match.

        Notes
        -----
        - X and y are reshaped to 2D: (n_samples, n_features) and (n_samples, n_targets).
        - Data is cast to float64 for numerical stability.
        """
        if not isinstance(X, tf.Tensor):
            raise TypeError("`X` must be a TensorFlow tensor.")
        if not isinstance(y, tf.Tensor):
            raise TypeError("`y` must be a TensorFlow tensor.")

        # Reshape to 2D if necessary
        X = tf.reshape(X, (-1, X.shape[-1]))
        y = tf.reshape(y, (-1, y.shape[-1] if y.ndim > 1 else 1))

        X = tf.cast(X, dtype=tf.float64)
        y = tf.cast(y, dtype=tf.float64)

        # Center the data
        X_mean = tf.reduce_mean(X, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # SVD of X_centered: X_centered = U * diag(s) * V^T
        s, U, Vt = tf.linalg.svd(X_centered, full_matrices=False)

        # Avoid division by zero for small singular values
        eps = tf.keras.backend.epsilon()
        threshold = eps * tf.reduce_max(s)

        # Apply ridge correction by shifting singular values
        # s_inv = 1 / (s + alpha)
        s_inv = tf.where(s > threshold, 1.0 / (s + self._alpha), 0.0)

        # Compute U^T * y_centered
        UTy = tf.matmul(U, y_centered, transpose_a=True)

        # Expand s_inv to a 2D for multiplication
        s_inv = s_inv[:, tf.newaxis]

        # Coefficients = V * diag(s_inv) * U^T * y_centered
        coef = tf.matmul(Vt, s_inv * UTy)

        # Intercept = y_mean - X_mean * coef
        intercept = y_mean - tf.matmul(X_mean, coef)

        self._coef = coef
        # Flatten intercept to shape (n_targets,)
        self._intercept = tf.reshape(intercept, [-1])

        self._built = True
        self._n_features_in = X.shape[-1]

        # Concatenate coef + intercept to facilitate prediction
        self._W = tf.concat([coef, intercept], axis=0)

        return self

    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Generate predictions using the fitted Ridge model.

        Parameters
        ----------
        X : tf.Tensor
            Input data of shape (n_samples, n_features).

        Returns
        -------
        tf.Tensor
            Predicted values, shape (n_samples, n_targets).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        TypeError
            If X is not a tf.Tensor.
        """
        if not self._built:
            raise ValueError("Model must be fitted before making predictions.")
        if not isinstance(X, tf.Tensor):
            raise TypeError("`X` must be a TensorFlow tensor.")

        # Reshape if necessary, cast to float64
        X = tf.reshape(X, (-1, X.shape[-1]))
        X = tf.cast(X, dtype=tf.float64)

        # Append a column of ones for intercept
        ones_column = tf.ones([X.shape[0], 1], dtype=tf.float64)
        X_bias = tf.concat([X, ones_column], axis=1)

        # (n_samples, n_features+1) x (n_features+1, n_targets) = (n_samples, n_targets)
        return tf.matmul(X_bias, self._W)

    @property
    def alpha(self) -> float:
        """Return the regularization strength."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """
        Set the regularization strength and invalidate current fit.

        Parameters
        ----------
        value : float
            New value for alpha. Must be >= 0.
        """
        if value < 0:
            raise ValueError("Regularization strength must be non-negative.")
        if not isinstance(value, (int, float)):
            raise TypeError("`alpha` must be an integer or float.")

        self._alpha = float(value)
        self._built = False
        self._coef = None
        self._intercept = None
        self._W = None
        self._n_features_in = None

    @property
    def built(self) -> bool:
        """Return whether the model has been fitted."""
        return self._built

    @property
    def coef_(self) -> Optional[tf.Tensor]:
        """
        Get the model coefficients (excluding intercept).

        Returns
        -------
        tf.Tensor or None
            Shape (n_features, n_targets) or None if not fitted.
        """
        return self._coef

    @property
    def intercept_(self) -> Optional[tf.Tensor]:
        """
        Get the model intercept.

        Returns
        -------
        tf.Tensor or None
            Shape (n_targets,) or None if not fitted.
        """
        return self._intercept

    @property
    def n_features_in(self) -> Optional[int]:
        """
        Number of features in the fitted model.

        Returns
        -------
        int or None
        """
        return self._n_features_in

    @property
    def W(self) -> Optional[tf.Tensor]:
        """
        Get the concatenated [coefficients, intercept] matrix.

        Returns
        -------
        tf.Tensor or None
            Shape (n_features+1, n_targets) or None if not fitted.
        """
        return self._W

    def get_params(self) -> Dict[str, Union[float, tf.Tensor, None]]:
        """
        Return a dictionary of model parameters.

        Returns
        -------
        dict
            Keys:
            - `alpha` : float
            - `coef_` : tf.Tensor or None
            - `intercept_` : tf.Tensor or None
        """
        return {
            "alpha": self._alpha,
            "coef_": self._coef,
            "intercept_": self._intercept,
        }


__all__ = [
    "create_tf_rng",
    "TF_Ridge",
]


def __dir__():
    return __all__
