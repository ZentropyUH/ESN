import os
import numpy as np
import pandas as pd
import xarray as xr


def load_csv(datapath: str) -> np.ndarray:
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = pd.read_csv(datapath, header=None)
    data = data.to_numpy()
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def load_npz(datapath: str) -> np.ndarray:
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = np.load(datapath)
    data = data["data"]
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def load_npy(datapath: str) -> np.ndarray:
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = np.load(datapath)
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def load_nc(datapath: str) -> np.ndarray:
    """Load the data from the given path. Returns a numpy array of shape (1, T, D)."""

    data = xr.open_dataarray(datapath)
    data = data.to_numpy()
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)

    return data


def save_csv(data: np.ndarray, savepath: str) -> None:
    """Save the data to the given path in .csv format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    df = pd.DataFrame(data=data)
    df.to_csv(savepath, index=False, header=False)


def save_npz(data: np.ndarray, savepath: str) -> None:
    """Save the data to the given path in .npz format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    np.savez(savepath, data=data)


def save_npy(data: np.ndarray, savepath: str) -> None:
    """Save the data to the given path in .npy format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    np.save(savepath, data)


def save_nc(data: np.ndarray, savepath: str) -> None:
    """Save the data to the given path in .nc format.

    Args:
        data (np.ndarray): The data to save.

        savepath (str): The path to save the data.
    """
    data = xr.DataArray(data=data)
    data.to_netcdf(savepath)


def list_files_only(directory: str) -> list:
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


def load_file(datapath: str) -> np.ndarray:
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


def load_data(
    datapath: str,
    init_transient: int = 0,
    transient: int = 1000,
    train_length: int = 5000,
    normalize: bool = False,
) -> tuple:
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


def save_data(
    data: np.ndarray, filename: str, savepath: str, format: str = "csv"
) -> None:
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
