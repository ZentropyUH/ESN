import numpy as np
import pandas as pd
import xarray as xr
import os
from natsort import natsorted
from scipy.stats import gmean


def list_files_only(directory):
    """
    List all files in a directory, excluding subdirectories.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of file names in the directory.
    """
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def load_file(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    if datapath.endswith(".csv"):
        data = load_csv(datapath)

    elif datapath.endswith(".npz"):
        data = load_npz(datapath)

    elif datapath.endswith(".npy"):
        data = load_npy(datapath)

    elif datapath.endswith(".nc"):
        data = load_nc(datapath)

    return data


def load_csv(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = pd.read_csv(datapath, header=None)
    data = data.to_numpy()
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def load_npz(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = np.load(datapath)
    data = data["data"]
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def load_npy(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = np.load(datapath)
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def load_nc(datapath):
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = xr.open_dataarray(datapath)
    data = data.to_numpy()
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def save_csv(data: np.ndarray, savepath: str):
    """Save the data to the given path in .csv format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    df = pd.DataFrame(data=data)
    df.to_csv(savepath, index=False, header=False)


def save_npz(data: np.ndarray, savepath: str):
    """Save the data to the given path in .npz format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    np.savez(savepath, data=data)


def save_npy(data: np.ndarray, savepath: str):
    """Save the data to the given path in .npy format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    np.save(savepath, data)


def save_nc(data: np.ndarray, savepath: str):
    """Save the data to the given path in .nc format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    data = xr.DataArray(data=data)
    data.to_netcdf(savepath)


def load_data(
    datapath: str,
    init_transient: int = 0,
    transient: int = 1000,
    train_length: int = 5000,
    normalize: bool = False,
):
    """
    Load data from a given path and prepare it for neural network training.

    Data is expected to be in the shape (1, T, D), where T is time and D is dimensions.

    Parameters
    ----------
    datapath : str
        Path to the data file. Supported formats: .csv, .npz, .npy, .nc.
    init_transient : int
        Number of initial transient time steps to discard before processing.
    transient : int, optional
        Length of the training transient for ESP, by default 1000.
    train_length : int, optional
        Length of the training data, by default 5000.
    normalize : bool, optional
        Whether to normalize the data, by default False.

    Returns
    -------
    tuple of np.ndarray
        Contains the following datasets:
        - transient_data: Transient portion of the training data (for ESP).
        - train_data: Training data.
        - train_target: Training target, shifted by one time step.
        - forecast_transient_data: Last 'transient' values of training data (for ESP).
        - val_data: Validation data.
        - val_target: Validation target, shifted by one time step.
    """
    print(f"Loading data from: {datapath}")

    data = load_file(datapath)
    _, T, D = data.shape

    if init_transient >= T:
        raise ValueError(
            f"init_transient ({init_transient}) is larger than data length ({T})."
        )

    # Trim initial transient
    data = data[:, init_transient:, :]
    T = data.shape[1]  # Update length after trimming

    train_index = transient + train_length

    if train_index > T:
        raise ValueError(
            f"Train size is out of range. Data shape after init_transient: {data.shape},"
            f" required train size + transient: {train_index}."
        )

    # Define data splits
    transient_data = data[:, :transient, :]
    train_data = data[:, transient:train_index, :]
    train_target = data[:, transient + 1 : train_index + 1, :]
    forecast_transient_data = train_data[:, -transient:, :]
    val_data = data[:, train_index:-1, :]
    val_target = data[:, train_index + 1 :, :]

    if normalize:
        mean = np.mean(train_data, axis=1, keepdims=True)
        std = np.std(train_data, axis=1, keepdims=True)

        train_data = (train_data - mean) / std
        train_target = (train_target - mean) / std
        val_data = (val_data - mean) / std
        val_target = (val_target - mean) / std
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


def save_data(data: np.ndarray, filename: str, savepath: str, format: str = "csv"):
    """Save the data to the given path.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.

        format (str, optional): The format to save the data. Defaults to "csv". Can be "csv", "npz", or "nc".
    """
    # squeeze data into a 2D array
    if data.ndim > 2:
        data = np.squeeze(data)

    full_path = os.path.join(savepath, filename + "." + format)

    if format == "csv":
        if data.ndim > 2:
            raise ValueError("Data must be 2D for csv format.")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        save_csv(data=data, savepath=full_path)
    elif format == "npz":
        save_npz(data=data, savepath=full_path)
    elif format == "npy":
        save_npy(data=data, savepath=full_path)
    elif format == "nc":
        save_nc(data=data, savepath=full_path)
        save_nc(data=data, savepath=full_path)


def compute_normalized_error(y_true, y_pred):
    """
    Compute a time-dependent, normalized RMS error for possibly multiple trajectories.

    Parameters
    ----------
    y_true : np.ndarray
        - If shape is (time, features), a single reference trajectory.
          Then all trajectories in y_pred (shape (samples, time, features))
          are compared against this single y_true.
        - If shape is (samples, time, features), multiple reference trajectories.
          Then y_pred must have the same shape, and y_true[i] is compared to y_pred[i].

    y_pred : np.ndarray
        Trajectories to compare, of shape (samples, time, features) or matching y_true if y_true has shape (samples, time, features).

    Returns
    -------
    final_error : np.ndarray
        1D array of length (time,). The normalized RMS error at each time step,
        averaged over all samples if multiple exist.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred.reshape(1, -1, y_pred.shape[-1])

    N_pred = y_pred.shape[0]

    if y_true.ndim == 2:
        y_true = np.concatenate([y_true[None, :, :]] * N_pred, axis=0)

    # -------------------------------
    # Both y_true and y_pred have equal shapes (samples, time, features)
    # -------------------------------
    # Here, y_true[i] is compared to y_pred[i].
    # We replicate the same logic on a per-sample basis:
    #   For each sample i:
    #     1) Compute a reference scale from y_true[i].
    #     2) Compute the time-dependent norm of (y_pred[i] - y_true[i]).
    #     3) Divide by that sample's reference scale.
    #   Finally, average across samples at each time step.

    if y_pred.shape != y_true.shape:
        raise ValueError(
            "When y_true has shape (samples, time, features), y_pred must match that shape."
        )

    samples = y_true.shape[0]

    # Prepare array for storing per-sample, time-dependent errors
    per_sample_errors = []

    for i in range(samples):
        # 1) reference scale for this sample
        ref_scale = np.sqrt(np.mean(y_true[i] ** 2, axis=0))  # (features,)
        ref_scale = np.linalg.norm(ref_scale)  # scalar

        # 2) difference -> shape (time, features)
        diff_i = y_pred[i] - y_true[i]
        # norm across features -> shape (time,)
        error_i = np.linalg.norm(diff_i, axis=1)

        # 3) normalize
        norm_error_i = error_i / ref_scale
        per_sample_errors.append(norm_error_i)

    # Stack -> shape (samples, time)
    per_sample_errors = np.array(per_sample_errors)

    # # Arithmetic mean
    # final_error = np.mean(per_sample_errors, axis=0)

    # Geometric mean
    final_error = gmean(per_sample_errors, axis=0)

    return final_error


def mean_ensemble_prediction(predictions):
    """
    Will average the predictions of all samples over the models axis.

    To explain in more detail, this will receive predictions of multiple models over multiple samples. The shape of the predictions is (models, samples, time, features), meaning that we have multiple models, each with multiple sample predictions over time and features. This function will average over the models axis, meaning that we will get the mean prediction over all models for each sample, over time and features.

    Args:
        predictions (np.ndarray): The predictions of shape (models, samples, time, features).

    Returns:
        np.ndarray: The mean predictions of shape (samples, time, features).
    """

    return np.mean(predictions, axis=0)


def get_all_predictions(predictions_path):
    """
    Will load all predictions from the given path and return them in a numpy array. Prediction files will always end with _predictions.[format], where [format] can be either npy, csv or nc. In the folder might be the target data as well, which will end with _target.[format].

    Args:
        predictions_path (str): The path to the directory containing the predictions.

    Returns:
        np.ndarray: The predictions of shape (models, samples, time, features).
    """

    # Get all files in the directory
    files = list_files_only(predictions_path)

    # Filter out the prediction files
    prediction_files = [f for f in files if "_predictions" in f]
    prediction_files = natsorted(prediction_files)

    # Load the first prediction to determine the shape
    first_prediction = load_file(os.path.join(predictions_path, prediction_files[0]))
    predictions_shape = (len(prediction_files),) + first_prediction.shape

    # Initialize the predictions array
    predictions = np.empty(predictions_shape, dtype=first_prediction.dtype)

    # Load the predictions into the array
    for i, file in enumerate(prediction_files):
        predictions[i] = load_file(os.path.join(predictions_path, file))

    return predictions


def get_all_targets(predictions_path):
    """
    Will load all target data from the given path and return them in a numpy array. Target files will always end with _target.[format], where [format] can be either npy, csv or nc. In the folder might be the predictions as well, which will end with _predictions.[format].

    Args:
        predictions_path (str): The path to the directory containing the targets.

    Returns:
        np.ndarray: The targets of shape (samples, time, features).
    """

    # Get all files in the directory
    files = list_files_only(predictions_path)

    # Filter out the target files
    target_files = [f for f in files if "_targets" in f]
    target_files = natsorted(target_files)

    # Load the first target to determine the shape
    first_target = load_file(os.path.join(predictions_path, target_files[0]))
    targets_shape = (len(target_files),) + first_target.shape

    # Initialize the targets array
    targets = np.empty(targets_shape, dtype=first_target.dtype)

    # Load the targets into the array
    for i, file in enumerate(target_files):
        targets[i] = load_file(os.path.join(predictions_path, file))

    return targets


def get_all_errors(predictions_path):
    """
    Given a path to a directory containing predictions and targets, this function will load all predictions and targets and compute the normalized error for each sample.

    Args:
        predictions_path (str): The path to the directory containing the predictions and targets.

    Returns:
        np.ndarray: The normalized errors of shape (samples, time).
    """

    preds = get_all_predictions(predictions_path)
    targets = get_all_targets(predictions_path)

    errors = np.empty((preds.shape[0], preds.shape[2], 1))

    for i, (pred, target) in enumerate(zip(preds, targets)):
        errors[i, :, 0] = compute_normalized_error(target, pred)

    return errors


def mean_prediction_error(predictions_path):
    """
    Given a datapath this will load all the predictions and take the mean over the models axis. We will then have the mean prediction for each sample. Then we will compare the mean prediction to the target data and return the normalized error.

    Args:
        predictions_path (str): The path to the directory containing the predictions.

    Returns:
        np.ndarray: The normalized error of shape (samples, time).
    """

    preds = get_all_predictions(predictions_path)
    mean_pred = mean_ensemble_prediction(preds)
    targets = get_all_targets(predictions_path)

    error = compute_normalized_error(targets, mean_pred)

    return error


__all__ = [
    # data_utils
    "list_files_only",
    "load_data",
    "save_data",
    "compute_normalized_error",
    "load_file",
    "mean_ensemble_prediction",
    "get_all_predictions",
    "get_all_targets",
    "get_all_errors",
]


def __dir__():
    return __all__
