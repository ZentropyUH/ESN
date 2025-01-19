"""Define some general utility functions."""

import os
import xarray as xr
from contextlib import contextmanager
from time import time

import keras
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D



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


def load_file(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    if datapath.endswith(".csv"):
        data = load_csv(datapath)

    elif datapath.endswith(".npz"):
        data = load_npz(datapath)
        
    elif datapath.endswith(".nc"):
        data = load_nc(datapath)


    return data

def load_csv(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = pd.read_csv(datapath, header=None)
    data = data.to_numpy()
    data = data.astype(np.float32)

    T, D = data.shape

    data = data.reshape(1, T, D)

    return data

def load_npz(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = np.load(datapath)
    data = data["data"]
    data = data.astype(np.float32)

    T, D = data.shape

    data = data.reshape(1, T, D)

    return data

def load_nc(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = xr.open_dataarray(datapath)
    data = data.to_numpy()
    data = data.astype(np.float32)

    T, D = data.shape

    data = data.reshape(1, T, D)

    return data

def load_data(
    datapath: str,
    transient: int = 1000,
    train_length: int = 5000,
    normalize: bool = False,
):
    """Load the data from the given path. Returns a dataset for training a NN.

    Data is supposed to be stored in a .csv and has a shape of (T, D), (T)ime and (D)imensions.

    Args:
        datapath (str): The datapath of the file to be loaded.

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

    data = load_file(datapath)

    _, T, D = data.shape

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
def timer(task_name, log=True):
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
    if log:
        print(f"\n{task_name}...\n")
    start = time()
    yield
    end = time()
    if log:
        print(f"{task_name} took: {round(end - start, 2)} seconds.\n")


def animate_trail(
    data,
    trail_length=50,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
    show=True,
    save_path=None,
    interval=15,
    dt=None,
):
    """
    Animate a point moving along the coordinates in x, y, and optionally z, leaving a trailing line.

    Args:
        data (np.ndarray): The data to animate. If 3D, it should have shape (T, 3), where T is the number of points. If 2D, it should have shape (T, 2).
        trail_length (int): Number of points to keep in the trail.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis (3D only).
        title (str): Title of the plot.
        show (bool): Whether to display the animation.
        save_path (str): Path to save the animation file, if specified.
        interval (int): Time in milliseconds between frames.
        dt (float): Time step between frames. If provided, it will be used to calculate the interval.
    """
    # Extract x, y, and z coordinates
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2] if data.shape[1] == 3 else None

    if dt is not None:
        interval = int(dt * 1000)

    # Set up the figure and axis, 3D if z is provided
    fig = plt.figure()
    if z is not None:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.set_ylim(min(y) - 1, max(y) + 1)
        ax.set_zlim(min(z) - 1, max(z) + 1)
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.set_ylim(min(y) - 1, max(y) + 1)

    # Set axis labels and title if provided
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if z is not None and zlabel:
        ax.set_zlabel(zlabel)

    # Initialize the point and trail line
    if z is not None:
        (point,) = ax.plot([], [], [], "bo", markersize=4)  # 3D point
        (trail_line,) = ax.plot([], [], [], "r-", alpha=0.7, linewidth=2)  # 3D trail
    else:
        (point,) = ax.plot([], [], "bo", markersize=4)  # 2D point
        (trail_line,) = ax.plot([], [], "r-", alpha=0.7, linewidth=2)  # 2D trail

    def init():
        """Initialize the animation with empty point and trail."""
        point.set_data([], [])
        trail_line.set_data([], [])
        if z is not None:
            point.set_3d_properties([])
            trail_line.set_3d_properties([])
        return point, trail_line

    def update(frame):
        """Update the point and trail for each frame."""
        # Update the main point position
        if z is not None:
            point.set_data([x[frame]], [y[frame]])
            point.set_3d_properties([z[frame]])
        else:
            point.set_data([x[frame]], [y[frame]])

        # Calculate the start of the trail
        start = max(0, frame - trail_length)

        # Update trail coordinates
        if z is not None:
            trail_line.set_data(x[start : frame + 1], y[start : frame + 1])
            trail_line.set_3d_properties(z[start : frame + 1])
        else:
            trail_line.set_data(x[start : frame + 1], y[start : frame + 1])

        return point, trail_line

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(x),
        init_func=init,
        blit=True,
        interval=interval,
        repeat=False,
    )

    # Display the animation
    if show:
        plt.show()

    # Save the animation
    if save_path is not None:
        # Calculate fps based on interval
        ani.save(save_path, writer="ffmpeg", fps=int(1000 / interval))


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
