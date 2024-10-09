"""Define some general utility functions."""

import numpy as np
import pandas as pd
import tensorflow as tf

from time import time
from contextlib import contextmanager


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


def load_data(
    name: str,
    transient: int = 1000,
    train_length: int = 5000,
    normalize: bool = False,
):
    """Load the data from the given path. Returns a dataset for training a NN.

    Data is supposed to be stored in a .csv and has a shape of (T, D), (T)ime and (D)imensions.

    Args:
        name (str): The name of the file to be loaded.

        transient (int, optional): The length of the training transient
                                    for teacher enforced process. Defaults to 1000.

        train_length (int, optional): The length of the training data. Defaults to 5000.

    Returns:
        tuple: A tuple with:

                transient_data: The transient of the training data. This is to ensure ESP.

                training_data: Training data.

                training_target: The training target. This is for forecasting, so target data is
                    the training data taken shifted 1 index to the right plus one value.

                forecast_transient_data: The last 'transient' elements in training_data.
                    This is to ensure ESP.

                validation_data: Validation data

                validation_target: The validation target. This is for forecasting, so target data is
                    the validation data taken shifted 1 index to the right plus one value.
    """
    data = pd.read_csv(name, header=None).to_numpy().astype(np.float32)

    T, D = data.shape
    data = data.reshape(1, T, D)


    # Index up to the training end.
    train_index = transient + train_length

    if train_index > T:
        raise ValueError(
            f"The train size is out of range. Data shape is: "
            f"{data.shape} and train size + transient is: {train_index}"
        )

    # Transient data (For ESP purposes)
    transient_data = data[:, :transient, :]

    train_data = data[:, transient:train_index, :]
    train_target = data[:, transient + 1 : train_index + 1, :]

    # Forecast transient (For ESP purposes).
    # These are the last 'transient' values of the training data
    forecast_transient_data = train_data[:, -transient:, :]

    val_data = data[:, train_index:-1, :]
    val_target = data[:, train_index + 1 :, :]

    if normalize:

        # Get the mean and std of the training data, over the time axis, independent for each batch (i.e. each element over the first axis)
        mean = np.mean(train_data, axis=1, keepdims=True)
        std = np.std(train_data, axis=1, keepdims=True)

        # Normalize the training data
        train_data = (train_data - mean) / std
        train_target = (train_target - mean) / std

        # Normalize the validation data
        val_data = (val_data - mean) / std
        val_target = (val_target - mean) / std

        # Normalize the transient data
        transient_data = (transient_data - mean) / std
        forecast_transient_data = (forecast_transient_data - mean) / std

    return (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    )


@contextmanager
def timer(task_name):
    """
    Context manager to measure the time of a task.

    Args:
        task_name (str): Name of the task to measure.

    Returns:
        None

    Example:
        >>> with self.timer("Task"):
        >>>     # Code to measure
        Will print the time taken to execute the code block.
    """
    print(f"\n{task_name}...\n")
    start = time()
    yield
    end = time()
    print(f"{task_name} took: {round(end - start, 2)} seconds.\n")


# Get the state of the ESN function
def get_esn_state(model):
    """Return the state of the ESN cell.

    Args:
        model (Model): The Keras model containing the ESN RNN layer.

    Returns:
        np array
    """
    # Access the ESN RNN layer by name and retrieve its last state
    esn_rnn_layer = model.get_layer("esn_rnn")
    state_h = esn_rnn_layer.states[0]

    # Convert the tensor to a NumPy array
    states = np.squeeze(state_h.numpy())

    return states


def calculate_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate the RMSE between the target and the prediction.

    Args:
        target (np array): The target data.

        prediction (np array): The prediction data.

    Returns:
        float: The RMSE between the target and the prediction.
    """
    return np.sqrt(np.mean(np.square(target - prediction)))


def calculate_nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate the NRMSE between the target and the prediction.

    Args:
        target (np array): The target data.

        prediction (np array): The prediction data.

    Returns:
        float: The NRMSE between the target and the prediction.
    """
    return calculate_rmse(target, prediction) / np.std(target)


def calculate_rmse_list(target: np.ndarray, prediction: np.ndarray):
    """
    Calculate the RMSE between the target and the prediction for a list of true and predicted values.

    Args:
        target (np array): The target data.

        prediction (np array): The prediction data.

    Returns:
        list: A list of RMSE values.
    """
    rmse_values = []
    for _target, _prediction in zip(target, prediction):
        rmse = calculate_rmse(_target, _prediction)
        rmse_values.append(rmse)
    return rmse_values


def calculate_nrmse_list(target: np.ndarray, prediction: np.ndarray):
    """
    Calculate the NRMSE between the target and the prediction for a list of true and predicted values.

    Args:
        target (np array): The target data.

        prediction (np array): The prediction data.

    Returns:
        list: A list of NRMSE values.
    """
    std = np.std(target)
    nrmse_values = []
    for _target, _prediction in zip(target, prediction):
        nrmse = np.sqrt(np.mean(np.square(_target - _prediction))) / std
        nrmse_values.append(nrmse)
    return nrmse_values


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
        self._intercept = tf.reshape(intercept, [-1]) # Remove the extra dimension of size 1
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
