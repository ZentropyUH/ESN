import numpy as np
import pandas as pd
import xarray as xr
import os


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

    # -------------------------------
    # Case 1: y_true has shape (time, features)
    # -------------------------------
    if y_true.ndim == 2:
        # We replicate the logic in your snippet exactly:
        # 1) Compute a single reference scale from y_true.
        # 2) For each trajectory in y_pred, compute the time-dependent norm of the difference.
        # 3) Divide by the reference scale.
        # 4) Average over trajectories (samples).

        # Ensure y_pred has shape (samples, time, features)
        if y_pred.ndim != 3:
            raise ValueError(
                "y_pred must have shape (samples, time, features) when y_true has shape (time, features)."
            )

        # 1) Reference scale is a single scalar
        #    sqrt(mean(y_true^2, axis=0)) -> (features,)
        #    then norm(...) -> scalar
        reference_scale = np.sqrt(np.mean(y_true**2, axis=0))  # shape (features,)
        reference_scale = np.linalg.norm(reference_scale)  # scalar

        # 2) Compute the time-dependent norm of (y_pred[i] - y_true), for each sample i
        #    y_pred has shape (samples, time, features)
        #    y_true has shape (time, features), so do y_true[None, :, :] => (1, time, features)
        #    Then broadcast
        diffs = y_pred - y_true[None, :, :]  # (samples, time, features)
        # Norm across features => shape (samples, time)
        error_by_time = np.linalg.norm(diffs, axis=2)

        # 3) Divide each time step by reference_scale => shape (samples, time)
        normalized_error = error_by_time / reference_scale

        # 4) Average across samples => shape (time,)
        final_error = np.mean(normalized_error, axis=0)
        return final_error

    # -------------------------------
    # Case 2: y_true has shape (samples, time, features)
    # -------------------------------
    elif y_true.ndim == 3:
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

        # Average across samples -> shape (time,)
        final_error = np.mean(per_sample_errors, axis=0)
        return final_error

    else:
        raise ValueError(
            "Invalid dimensions for y_true. Expected (time, features) or (samples, time, features). "
            f"Got shape {y_true.shape}."
        )
