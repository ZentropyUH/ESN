import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Tuple


def load_csv(datapath: str) -> np.ndarray:
    """
    Load CSV data from the specified path and return it as a NumPy array with shape (1, T, D).

    This function ensures that the returned data has a "batch" dimension of size 1 if
    the original data is 2D.

    Parameters
    ----------
    datapath : str
        Path to the .csv file.

    Returns
    -------
    np.ndarray
        A float32 NumPy array of shape (1, T, D).

    Raises
    ------
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data = pd.read_csv(datapath, header=None).to_numpy()
    data = data.astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"CSV data has an unsupported shape {data.shape}. Expected 2D or 3D."
        )

    return data


def load_npz(datapath: str) -> np.ndarray:
    """
    Load data from a .npz file and return it as a NumPy array with shape (1, T, D).

    Expects the .npz file to contain an array under the key "data".

    Parameters
    ----------
    datapath : str
        Path to the .npz file.

    Returns
    -------
    np.ndarray
        A float32 NumPy array of shape (1, T, D).

    Raises
    ------
    KeyError
        If "data" key is not found in the .npz file.
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data_dict = np.load(datapath)
    if "data" not in data_dict:
        raise KeyError(f"Key 'data' not found in the .npz file: {datapath}.")

    data = data_dict["data"].astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"NPZ data has an unsupported shape {data.shape}. Expected 2D or 3D."
        )

    return data


def load_npy(datapath: str) -> np.ndarray:
    """
    Load data from a .npy file and return it as a NumPy array with shape (1, T, D).

    Parameters
    ----------
    datapath : str
        Path to the .npy file.

    Returns
    -------
    np.ndarray
        A float32 NumPy array of shape (1, T, D).

    Raises
    ------
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data = np.load(datapath).astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"NPY data has an unsupported shape {data.shape}. Expected 2D or 3D."
        )

    return data


def load_nc(datapath: str) -> np.ndarray:
    """
    Load data from a NetCDF (.nc) file and return it as a NumPy array with shape (1, T, D).

    Parameters
    ----------
    datapath : str
        Path to the .nc file.

    Returns
    -------
    np.ndarray
        A float32 NumPy array of shape (1, T, D).

    Raises
    ------
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data = xr.open_dataarray(datapath).to_numpy().astype(np.float32)

    if data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"NetCDF data has an unsupported shape {data.shape}. Expected 2D or 3D."
        )

    return data


def save_csv(data: np.ndarray, savepath: str) -> None:
    """
    Save the given data to a .csv file.

    Parameters
    ----------
    data : np.ndarray
        Data to save. Must be 2D.
    savepath : str
        Path (including filename) to save the .csv file.

    Raises
    ------
    ValueError
        If the data is not 2D.
    """
    if data.ndim != 2:
        raise ValueError("CSV saving requires a 2D array.")
    df = pd.DataFrame(data=data)
    df.to_csv(savepath, index=False, header=False)


def save_npz(data: np.ndarray, savepath: str) -> None:
    """
    Save the given data to a .npz file using the "data" key.

    Parameters
    ----------
    data : np.ndarray
        Data to save.
    savepath : str
        Path (including filename) to save the .npz file.
    """
    np.savez(savepath, data=data)


def save_npy(data: np.ndarray, savepath: str) -> None:
    """
    Save the given data to a .npy file.

    Parameters
    ----------
    data : np.ndarray
        Data to save.
    savepath : str
        Path (including filename) to save the .npy file.
    """
    np.save(savepath, data)


def save_nc(data: np.ndarray, savepath: str) -> None:
    """
    Save the given data to a NetCDF (.nc) file.

    Parameters
    ----------
    data : np.ndarray
        Data to save.
    savepath : str
        Path (including filename) to save the .nc file.
    """
    data_array = xr.DataArray(data=data)
    data_array.to_netcdf(savepath)


def list_files_only(directory: str) -> List[str]:
    """
    List all files (not directories) in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory.

    Returns
    -------
    List[str]
        A list of file names in the directory, excluding subdirectories.
    """
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def load_file(datapath: str) -> np.ndarray:
    """
    Load data from a file path with one of the supported extensions: .csv, .npz, .npy, .nc.

    The returned data will have the shape (1, T, D).

    Parameters
    ----------
    datapath : str
        Path to the data file.

    Returns
    -------
    np.ndarray
        Data loaded from the file, with shape (1, T, D).

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    if datapath.endswith(".csv"):
        data = load_csv(datapath)
    elif datapath.endswith(".npz"):
        data = load_npz(datapath)
    elif datapath.endswith(".npy"):
        data = load_npy(datapath)
    elif datapath.endswith(".nc"):
        data = load_nc(datapath)
    else:
        raise ValueError(
            f"Unsupported file extension for datapath: {datapath}. "
            "Supported extensions are .csv, .npz, .npy, and .nc."
        )
    return data


def load_data(
    datapath: str,
    init_transient: int = 0,
    transient: int = 1000,
    train_length: int = 5000,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare time-series data for neural network (e.g., ESN/Reservoir) training.

    The data is assumed to be of shape (1, T, D), where T is time steps, and D is the dimension.

    Parameters
    ----------
    datapath : str
        Path to the data file. Supported formats: .csv, .npz, .npy, .nc.
    init_transient : int, optional
        Number of initial transient time steps to discard, by default 0.
    transient : int, optional
        Length of the transient period for ESN/Reservoir training, by default 1000.
    train_length : int, optional
        Length of the training data (after the transient), by default 5000.
    normalize : bool, optional
        Whether to normalize the data (based on mean and std of the training portion), by default False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - transient_data (np.ndarray):  The initial portion (transient length) of the training data.
        - train_data (np.ndarray):      The training data sequence.
        - train_target (np.ndarray):    The training target, time-shifted by one step relative to train_data.
        - forecast_transient_data (np.ndarray): The last 'transient' steps of the training data for forecast initialization.
        - val_data (np.ndarray):        The validation data sequence following the training portion.
        - val_target (np.ndarray):      The validation target, time-shifted by one step relative to val_data.

    Raises
    ------
    ValueError
        If init_transient is >= the total time steps T.
        If the required train size (transient + train_length) exceeds the available time steps.
    """
    print(f"Loading data from: {datapath}")
    data = load_file(datapath)
    _, T, D = data.shape

    if init_transient >= T:
        raise ValueError(
            f"init_transient ({init_transient}) is larger or equal to total data length ({T})."
        )

    # Trim initial transient
    data = data[:, init_transient:, :]
    T = data.shape[1]  # Update length after trimming

    train_index = transient + train_length
    if train_index > T:
        raise ValueError(
            f"Train size (transient + train_length = {train_index}) exceeds data length ({T})."
        )

    # Define data splits
    transient_data = data[:, :transient, :]
    train_data = data[:, transient:train_index, :]
    train_target = data[:, transient + 1 : train_index + 1, :]
    forecast_transient_data = train_data[:, -transient:, :]
    val_data = data[:, train_index:-1, :]
    val_target = data[:, train_index + 1 :, :]

    # Optional normalization
    if normalize:
        mean = np.mean(train_data, axis=1, keepdims=True)
        std = np.std(train_data, axis=1, keepdims=True)

        # Prevent division by zero
        std = np.where(std == 0, 1, std)

        transient_data = (transient_data - mean) / std
        train_data = (train_data - mean) / std
        train_target = (train_target - mean) / std
        forecast_transient_data = (forecast_transient_data - mean) / std
        val_data = (val_data - mean) / std
        val_target = (val_target - mean) / std

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
    """
    Save the data to the specified path in the chosen format.

    Parameters
    ----------
    data : np.ndarray
        The data to save. Can be 1D, 2D, or 3D.
    filename : str
        The base filename (without extension).
    savepath : str
        The directory in which to save the file.
    format : str, optional
        The format to use when saving. Options: "csv", "npz", "npy", "nc".
        Defaults to "csv".

    Raises
    ------
    ValueError
        If format is unknown or if attempting to save a multidimensional array (ndim > 2)
        in CSV format.
    """
    # Squeeze data into at most 2D if possible, to handle CSV case
    data = np.squeeze(data)
    full_path = os.path.join(savepath, filename + "." + format)

    if format == "csv":
        # CSV requires 2D data
        if data.ndim > 2:
            raise ValueError("Data must be 2D or lower to be saved as CSV.")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        save_csv(data=data, savepath=full_path)
    elif format == "npz":
        save_npz(data=data, savepath=full_path)
    elif format == "npy":
        save_npy(data=data, savepath=full_path)
    elif format == "nc":
        save_nc(data=data, savepath=full_path)
    else:
        raise ValueError(
            f"Unknown format: {format}. Supported formats are 'csv', 'npz', 'npy', 'nc'."
        )
