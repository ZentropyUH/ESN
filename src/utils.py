"""Define some general utility functions."""
import numpy as np
import pandas as pd

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
    step: int = 1,
):
    """Load the data from the given path. Returns a dataset for training a NN.

    Data is supposed to be stored in a .csv and has a shape of (T, D), (T)ime and (D)imensions.

    Args:
        name (str): The name of the file to be loaded.

        transient (int, optional): The length of the training transient
                                    for teacher enforced process. Defaults to 1000.

        train_length (int, optional): The length of the training data. Defaults to 5000.

        step: Sets the number of steps between data sampling. i. e. takes values every 'step' steps

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
    data = pd.read_csv(name).to_numpy()

    features = data.shape[-1]

    data = data[::step]

    data = data.reshape(1, -1, features)

    # Take the elements of the data skipping every step elements.

    if step > 1:
        print(
            "Used data shape: ",
            data.shape,
            f"Picking values every {step} steps.",
        )

    # Index up to the training end.
    train_index = transient + train_length

    if train_index > data.shape[1]:
        raise ValueError(
            f"The train size is out of range. Data size is: "
            f"{data.shape[0]} and train size + transient is: {train_index}"
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
    return np.sqrt(np.mean(np.square(target - prediction))) / np.std(target)


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
