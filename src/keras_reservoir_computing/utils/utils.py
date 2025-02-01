"""Define some general utility functions."""

from contextlib import contextmanager
from time import time
import tensorflow as tf



# given i it starts from letter x and goes cyclically, when x reached starts xx, xy, etc.
def letter(n: int) -> str:
    """Return the letter corresponding to the given number.

    Args:
        n (int): The number to convert to a letter. Starting from letter x, it goes cyclically adding a letter x to the left.

    Returns:
        str: The letter corresponding to the given number. Before 26, would be equivalent to chr(n + 97 + 23).
    """
    return "x" * ((n + 23) // 26) + chr(ord("a") + (n + 23) % 26)


def lyap_ks(i_th, L_period):
    """Estimation of the i-th largest Lyapunov Time of the KS model.

    Args:
        i_th (int): The i-th largest Lyapunov Time.

        L_period (int): The period of the system.

    Returns:
        float: The estimated i-th largest Lyapunov Time.

    Taken from the paper:
        "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arxiv:1902.09651v1"
    """
    # This approximation is taken from the above paper. Verify veracity.
    return 0.093 - 0.94 * (i_th - 0.39) / L_period


@contextmanager
def timer(task_name: str="Task", log: bool=True):
    """
    Context manager to measure the time of a task.

    Args:
        task_name (str): Name of the task to measure.
        log (bool): Whether to log the time taken.

    Returns:
        None

    Example:
        >>> with self.timer("Some Task"):
        >>>     # Code to measure
        Will print the time taken to execute the code block.
    """
    if log:
        print(f"\n{task_name}...\n")
    start = time()
    yield
    end = time()
    if log:
        print(f"{task_name} took: {round(end - start, 2)} seconds.\n")


# TF implementation of Ridge using svd. TODO: see if it works as well as sklearn
class TF_Ridge:
    """
    Robust tensorflow ridge regression model using SVD solver.

    Args:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float) -> None:
        if alpha < 0:
            raise ValueError("Regularization strength must be non-negative.")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be an integer or float.")
        self._alpha = alpha
        self._built = False
        self._coef = None
        self._intercept = None
        self._W = None

    def fit(self, X, y):
        """
        Fit the Ridge regression model using SVD. It uses the formula:

        W = (X^T X + alpha I)^-1 X^T y

        Which, when using SVD, translates computationally to:

        W = V S^-1 U^T y

        Where:
        - V is the matrix of right singular vectors of X
        - S is the diagonal matrix of singular values of X
        - S^-1 = 1/(S + alpha) is the diagonal matrix of the inverse of the singular values of X
        - U is the matrix of left singular vectors of X
        - y is the target data


        Args:
            X (tf.Tensor): The input data.

            y (tf.Tensor): The target data.

        Returns:
            Ridge: The fitted Ridge regression model.
        """
        if not isinstance(X, tf.Tensor):
            raise TypeError("X must be a TensorFlow tensor.")
        if not isinstance(y, tf.Tensor):
            raise TypeError("y must be a TensorFlow tensor.")

        X = tf.reshape(X, (-1, X.shape[-1]))
        y = tf.reshape(y, (-1, y.shape[-1]))

        X = tf.cast(X, dtype=tf.float64)
        y = tf.cast(y, dtype=tf.float64)

        # Center the data
        X_mean = tf.reduce_mean(X, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Compute SVD of X_centered
        s, U, Vt = tf.linalg.svd(X_centered, full_matrices=False)

        # Avoid division by zero for small singular values.
        tol = tf.keras.backend.epsilon()

        # Normalize with the maximum singular value,
        # that is the one with most information
        threshold = tol * tf.reduce_max(s)
        s_inv = tf.where(s > threshold, 1 / (s + self._alpha), 0)

        # Compute U^T y_centered
        UTy = tf.matmul(U, y_centered, transpose_a=True)

        # Expand s_inv to a matrix
        s_inv = s_inv[:, tf.newaxis]

        # Compute coefficients
        coef = tf.matmul(Vt, s_inv * UTy)

        # Compute the intercept
        intercept = y_mean - tf.matmul(X_mean, coef)

        self._coef = coef
        self._intercept = tf.reshape(
            intercept, [-1]
        )  # Remove the extra dimension of size 1
        self._built = True
        self._n_features_in = X.shape[-1]
        self._W = tf.concat([coef, intercept], axis=0)

        return self

    def predict(self, X):
        """
        Predict using the Ridge regression model.

        Args:
            X (tf.Tensor): The input data.

        Returns:
            tf.Tensor: The predicted values.
        """
        if not self._built:
            raise ValueError("Model must be fitted before making predictions.")
        if not isinstance(X, tf.Tensor):
            raise TypeError("X must be a TensorFlow tensor.")

        X = tf.reshape(X, (-1, X.shape[-1]))
        X = tf.cast(X, dtype=tf.float64)

        # Add a column of ones to X for the intercept
        X = tf.concat([X, tf.ones((X.shape[0], 1), dtype=tf.float64)], axis=1)

        # Compute predictions
        predictions = tf.matmul(X, self._W)

        return predictions

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("Regularization strength must be non-negative.")
        if not isinstance(value, (int, float)):
            raise TypeError("alpha must be an integer or float.")
        self._alpha = value
        self._built = False
        self._coef = None
        self._intercept = None
        self._n_features_in = None
        self._W = None

    @property
    def built(self):
        return self._built

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @property
    def n_features_in(self):
        return self._n_features_in

    @property
    def W(self):
        return self._W

    def get_params(self):
        """
        Get the parameters of the Ridge regression model.

        Returns:
            dict: Dictionary containing 'coef_' and 'intercept_'.
        """
        return {
            "alpha": self._alpha,
            "coef_": self.coef_,
            "intercept_": self.intercept_,
        }
