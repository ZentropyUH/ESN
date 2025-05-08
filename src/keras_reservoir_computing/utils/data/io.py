import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


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
        A NumPy array of shape (1, T, D).

    Raises
    ------
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data = pd.read_csv(datapath, header=None).to_numpy()

    if data.ndim == 1:
        T = data.shape[0]
        data = data.reshape(1, T, 1)
    elif data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"CSV data has an unsupported shape {data.shape}. Expected 1D, 2D or 3D."
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
        A NumPy array of shape (1, T, D).

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

    data = data_dict["data"]

    if data.ndim == 1:
        T = data.shape[0]
        data = data.reshape(1, T, 1)
    elif data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"CSV data has an unsupported shape {data.shape}. Expected 1D, 2D or 3D."
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
        A NumPy array of shape (1, T, D).

    Raises
    ------
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data = np.load(datapath)

    if data.ndim == 1:
        T = data.shape[0]
        data = data.reshape(1, T, 1)
    elif data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"CSV data has an unsupported shape {data.shape}. Expected 1D, 2D or 3D."
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
        A NumPy array of shape (1, T, D).

    Raises
    ------
    ValueError
        If the data cannot be reshaped to (1, T, D).
    """
    data = xr.open_dataarray(datapath).to_numpy()

    if data.ndim == 1:
        T = data.shape[0]
        data = data.reshape(1, T, 1)
    elif data.ndim == 2:
        T, D = data.shape
        data = data.reshape(1, T, D)
    elif data.ndim != 3:
        raise ValueError(
            f"CSV data has an unsupported shape {data.shape}. Expected 1D, 2D or 3D."
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
    datapath: Union[str, List[str]],
    init_transient: int = 0,
    transient_length: int = 1000,
    train_length: int = 5000,
    val_length: int = 5000,
    normalize: bool = False,
    normalization_method: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare time-series data for neural network (e.g., ESN/Reservoir) training.

    The data is assumed to be of shape (B, T, D), where T is time steps, and D is the dimension.

    Parameters
    ----------
    datapath : str or List[str]
        Path to the data file. Supported formats: .csv, .npz, .npy, .nc. If a list of paths is
        provided, the data will be concatenated along the batch dimension.
    init_transient : int, optional
        Number of initial transient time steps to discard, by default 0.
    transient_length : int, optional
        Length of the transient period for ESN/Reservoir training, by default 1000.
    train_length : int, optional
        Length of the training data (after the transient), by default 5000.
    val_length : int, optional
        Length of the validation data. If None, uses all remaining data after training, by default None.
    normalize : bool, optional
        Whether to normalize the data (based on mean and std of the training portion), by default False.
    normalization_method : str, optional
        The method to use for normalization. Options: "standard", "minmax".
        By default "standard".

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
        If the required train+val size exceeds the available time steps.
    """

    if isinstance(datapath, list):
        data = np.concatenate([load_file(f) for f in datapath], axis=0)
    else:
        data = load_file(datapath)

    B, T, D = data.shape

    if init_transient >= T:
        raise ValueError(
            f"init_transient ({init_transient}) is larger or equal to total data length ({T})."
        )

    # Trim initial transient
    data = data[:, init_transient:, :]
    T = data.shape[1]  # Update T after trimming initial transient

    train_index = transient_length + train_length
    if train_index > T:
        raise ValueError(
            f"Train size (transient + train_length = {train_index}) exceeds data length ({T})."
        )

    # If val_length is None, use all remaining data, otherwise use specified length
    if val_length is None:
        val_length = T - train_index - 1  # -1 because we need targets (shifted by 1)
    elif train_index + val_length + 1 > T:  # +1 for targets
        raise ValueError(
            f"Combined data requirements (train + val + targets = {train_index + val_length + 1}) "
            f"exceed available data length ({T})."
        )

    # Define data splits
    transient_data = data[:, :transient_length, :]
    train_data = data[:, transient_length:train_index, :]
    train_target = data[:, transient_length + 1 : train_index + 1, :]
    ftransient = train_data[:, -transient_length:, :]
    val_data = data[:, train_index:train_index + val_length, :]
    val_target = data[:, train_index + 1:train_index + val_length + 1, :]

    # Optional normalization
    if normalize:
        if normalization_method == "standard":
            # Normalize to zero mean and unit variance
            mean = np.mean(train_data, axis=1, keepdims=True)
            std = np.std(train_data, axis=1, keepdims=True)

            # Prevent division by zero
            std = np.where(std == 0, 1, std)

            transient_data = (transient_data - mean) / std
            train_data = (train_data - mean) / std
            train_target = (train_target - mean) / std
            ftransient = (ftransient - mean) / std
            val_data = (val_data - mean) / std
            val_target = (val_target - mean) / std
        elif normalization_method == "minmax":
            # Normalize to [-1, 1]
            _min = np.min(train_data, axis=1, keepdims=True)
            _max = np.max(train_data, axis=1, keepdims=True)

            # Prevent division by zero
            div = _max - _min
            div = np.where(div == 0, 1, div)

            transient_data = 2 * (transient_data - _min) / div - 1
            train_data = 2 * (train_data - _min) / div - 1
            train_target = 2 * (train_target - _min) / div - 1
            ftransient = 2 * (ftransient - _min) / div - 1
            val_data = 2 * (val_data - _min) / div - 1
            val_target = 2 * (val_target - _min) / div - 1

        else:
            raise ValueError(
                f"Unknown normalization method: {normalization_method}. "
                "Supported methods are 'standard' and 'minmax'."
            )

    return (
        transient_data,
        train_data,
        train_target,
        ftransient,
        val_data,
        val_target,
    )


def load_data_dual(
    train_path: Union[str, List[str]],
    val_path: Union[str, List[str]],
    init_transient: int = 0,
    transient_length: int = 1000,
    train_length: int = 5000,
    val_length: int = 5000,
    normalize: bool = False,
    normalization_method: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and validation data from two distinct sources (files or lists of files),
    handling initial transient trimming and normalization. Returns properly aligned
    transient/train/val datasets for ESN workflows.

    Parameters
    ----------
    train_path : str or List[str]
        Path(s) to training data file(s).
    val_path : str or List[str]
        Path(s) to validation data file(s).
    init_transient : int
        Number of initial time steps to discard in all files.
    transient_length : int
        Length of the transient data used before training.
    train_length : int
        Length of the training sequence.
    val_length : int
        Length of the validation sequence.
    normalize : bool
        Whether to normalize the data based on training stats.
    normalization_method : str
        Normalization method: "standard" or "minmax".

    Returns
    -------
    Tuple of 6 arrays:
        transient_data, train_data, train_target,
        ftransient_data, val_data, val_target
    """
    if isinstance(train_path, list) and isinstance(val_path, list):
        if len(train_path) != len(val_path):
            raise ValueError("Train and validation file lists must have the same length.")
        train_batches = [load_file(p) for p in train_path]
        val_batches = [load_file(p) for p in val_path]
        train_data_all = np.concatenate(train_batches, axis=0)
        val_data_all = np.concatenate(val_batches, axis=0)
    elif isinstance(train_path, str) and isinstance(val_path, str):
        train_data_all = load_file(train_path)
        val_data_all = load_file(val_path)
    else:
        raise TypeError("train_path and val_path must both be str or both be List[str]")

    # Discard system-level initial transient
    train_data_all = train_data_all[:, init_transient:, :]
    val_data_all = val_data_all[:, init_transient:, :]

    T_train = train_data_all.shape[1]
    T_val = val_data_all.shape[1]

    if transient_length + train_length + 1 > T_train:
        raise ValueError("Train file(s) too short for requested transient + train + target.")
    if transient_length + val_length + 1 > T_val:
        raise ValueError("Val file(s) too short for requested transient + val + target.")

    # TRAIN SPLITS
    transient_data = train_data_all[:, :transient_length, :]
    train_data = train_data_all[:, transient_length:transient_length + train_length, :]
    train_target = train_data_all[:, transient_length + 1:transient_length + train_length + 1, :]

    # VAL SPLITS
    ftransient_data = val_data_all[:, :transient_length, :]
    val_data = val_data_all[:, transient_length:transient_length + val_length, :]
    val_target = val_data_all[:, transient_length + 1:transient_length + val_length + 1, :]

    if normalize:
        if normalization_method == "standard":
            mean = np.mean(train_data, axis=1, keepdims=True)
            std = np.std(train_data, axis=1, keepdims=True)
            std = np.where(std == 0, 1, std)
            transient_data = (transient_data - mean) / std
            train_data = (train_data - mean) / std
            train_target = (train_target - mean) / std
            ftransient_data = (ftransient_data - mean) / std
            val_data = (val_data - mean) / std
            val_target = (val_target - mean) / std
        elif normalization_method == "minmax":
            _min = np.min(train_data, axis=1, keepdims=True)
            _max = np.max(train_data, axis=1, keepdims=True)
            div = np.where(_max - _min == 0, 1, _max - _min)
            transient_data = 2 * (transient_data - _min) / div - 1
            train_data = 2 * (train_data - _min) / div - 1
            train_target = 2 * (train_target - _min) / div - 1
            ftransient_data = 2 * (ftransient_data - _min) / div - 1
            val_data = 2 * (val_data - _min) / div - 1
            val_target = 2 * (val_target - _min) / div - 1
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

    return (
        transient_data,
        train_data,
        train_target,
        ftransient_data,
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
