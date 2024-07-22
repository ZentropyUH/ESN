"""Define some general utility functions."""
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List


# given i it starts from letter x and goes cyclically, when x reached starts xx, xy, etc.
letter = lambda n: 'x' * ((n + 23) // 26) + chr(ord('a') + (n + 23) % 26)

def lyap_ks(i, l):
    """Estimation of the i-th largest Lyapunov Time of the KS model.

    Taken from the paper:
        "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arxiv:1902.09651v1"
    """
    # This approximation is taken from the above paper. Verify veracity.
    return 0.093 - 0.94 * (i - 0.39) / l


def load_data(
    name: str,
    transient: int = 1000,
    train_length: int = 5000,
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
    data = pd.read_csv(name, header=None).to_numpy()    

    data = data.astype(np.float32)

    features = data.shape[-1]

    data = data.reshape(1, -1, features)

    # Index up to the training end.
    train_index = transient + train_length

    if train_index > data.shape[1]:
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

    return (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    )


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

#TF implementation of Ridge using svd. TODO: see if it works as well as sklearn
class TF_RidgeRegression:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._coef = None
        self._intercept = None
        self.W_ = None

    def fit(self, X: tf.Tensor, y: tf.Tensor) -> 'TF_RidgeRegression':
        """
        Fit Ridge regression model using SVD, handling multi-dimensional y.

        Args:
            X (tf.Tensor): Input data of shape (n_samples, n_features).
            y (tf.Tensor): Target data of shape (n_samples, n_targets).

        Returns:
            self: Fitted model.
        """
        
        print("WHAT THE FUCK: ", X.dtype)
        print("WHAT THE FUCKING FUCK: ", y.dtype)
        
        assert X.ndim == 2, "X must be a 2D tensor."
        assert y.ndim == 2, "y must be a 2D tensor."
        
        n_samples, n_features = X.shape
        
        # Compute mean and standard deviation of X and y
        self._X_mean = tf.reduce_mean(X, axis=0)
        self._X_std = tf.math.reduce_std(X, axis=0)
        self._y_mean = tf.reduce_mean(y)

        # Center and scale X
        X_centered = (X - self._X_mean) / self._X_std
        y_centered = y - self._y_mean

        # Add a column of ones to the centered data to include the intercept term
        ones = tf.ones((n_samples, 1), dtype=tf.float32)
        X_centered_bias = tf.concat([X_centered, ones], axis=1)

        # Perform SVD on the centered data
        S, U, Vt = tf.linalg.svd(X_centered_bias, full_matrices=False)

        # Diagonal matrix of singular values with regularization
        D_inv = tf.linalg.diag(1 / (S**2 + self._alpha))

        # Calculate the coefficients using the SVD components
        Ut_y = tf.matmul(tf.transpose(U), y_centered)
        V_D_inv = tf.matmul(Vt, D_inv)
        self.W_ = tf.matmul(V_D_inv, tf.matmul(tf.linalg.diag(S), Ut_y))
        
        # Extract coefficients
        self._coef = self.W_[:-1]  # Coefficients (excluding bias term)

        # Adjust intercept to incorporate the mean values for each target
        self._coef = self._coef * self._X_std
        self._intercept = self._y_mean - tf.tensordot(self._X_mean, self.W_[:-1], axes=1) + self.W_[-1]
        
        # store the intercept in the last row of W_
        self.W_ = tf.concat([self.W_[:-1], tf.reshape(self._intercept, (1, -1))], axis=0)

        return self

    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Predict using the Ridge regression model, using adjusted intercepts for each target.

        Args:
            X (tf.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            tf.Tensor: Predicted values of shape (n_samples, n_targets).
        """
        if self.W_ is None:
            raise ValueError("Model has not been fitted yet.")

        # Add a column of ones to include the intercept term
        ones = tf.ones((tf.shape(X)[0], 1), dtype=tf.float32)
        X_bias = tf.concat([X, ones], axis=1)

        return tf.matmul(X_bias, self.W_)

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("Regularization strength must be non-negative.")        
        self._alpha = value
    
    @property
    def coef_(self):
        return self._coef
        
    @property
    def intercept_(self):
        return self._intercept
    
    def get_params(self):
        """
        Get the parameters of the Ridge regression model.

        Returns:
            dict: Dictionary containing 'coef_' and 'intercept_'.
        """
        return {'alpha': self._alpha, 'coef_': self.coef_, 'intercept_': self.intercept_}