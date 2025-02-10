import os

import numpy as np
from natsort import natsorted
from scipy.stats import gmean

from .io import list_files_only, load_file


def compute_normalized_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the time-dependent normalized root mean square (RMS) error between predicted and true trajectories.

    This function supports single and multiple trajectory comparisons. If `y_true` contains a single trajectory,
    all trajectories in `y_pred` are compared to it. If `y_true` contains multiple trajectories, `y_pred` must have
    the same shape, and comparisons are made pairwise.

    Parameters
    ----------
    y_true : np.ndarray
        Reference trajectories with shape:
        - (time, features): A single reference trajectory. All `y_pred` samples will be compared against this.
        - (samples, time, features): Multiple reference trajectories, where each `y_true[i]` is compared to `y_pred[i]`.

    y_pred : np.ndarray
        Predicted trajectories with shape:
        - (samples, time, features) if `y_true` contains multiple trajectories.
        - (time, features) or (samples, time, features) if `y_true` contains a single trajectory.

    Returns
    -------
    final_error : np.ndarray
        A 1D array of shape `(time,)`, representing the normalized RMS error at each time step,
        averaged over all samples when applicable.

    Raises
    ------
    ValueError
        If `y_true` and `y_pred` have incompatible shapes.

    Notes
    -----
    - The normalization scale is computed per sample using the RMS of `y_true[i]` over features.
    - Errors are computed as the geometric mean across samples at each time step.

    Example
    -------
    >>> import numpy as np
    >>> y_true = np.array([
    ...     [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # Sample 1
    ...     [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]   # Sample 2
    ... ])
    >>> y_pred = np.array([
    ...     [[1.1, 2.1], [1.9, 3.1], [3.1, 3.9]],  # Slightly deviating predictions
    ...     [[1.3, 2.4], [2.8, 3.4], [3.4, 4.6]]
    ... ])
    >>> compute_normalized_error(y_true, y_pred)
    array([0.043, 0.051, 0.034])    # Example output, values will vary

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[None, :, :]

    N_pred = y_pred.shape[0]

    if y_true.ndim == 2 or y_true.shape[0] == 1:
        y_true = np.tile(y_true.reshape(1, *y_true.shape[-2:]), (N_pred, 1, 1))

    if y_pred.shape != y_true.shape:
        raise ValueError(
            "When y_true has shape (samples, time, features), y_pred must match that shape."
        )

    # Compute reference scale per sample
    ref_scale = np.linalg.norm(
        np.sqrt(np.mean(y_true**2, axis=1)), axis=1, keepdims=True
    )

    # Compute error norms
    diff = np.linalg.norm(y_pred - y_true, axis=2)

    # Normalize
    norm_error = diff / ref_scale  # Shape (samples, time)

    # Compute geometric mean across samples
    final_error = gmean(norm_error, axis=0)

    return final_error


def mean_ensemble_prediction(predictions: np.ndarray) -> np.ndarray:
    """
    Will average the predictions of all samples over the models axis.

    To explain in more detail, this will receive predictions of multiple models over multiple samples. The shape of the predictions is (models, samples, time, features), meaning that we have multiple models, each with multiple sample predictions over time and features. This function will average over the models axis, meaning that we will get the mean prediction over all models for each sample, over time and features.

    Args:
        predictions (np.ndarray): The predictions of shape (models, samples, time, features).

    Returns:
        np.ndarray: The mean predictions of shape (samples, time, features).
    """

    return np.mean(predictions, axis=0)


def get_all_predictions(predictions_path: str) -> np.ndarray:
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


def get_all_targets(predictions_path: str) -> np.ndarray:
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


def get_all_errors(predictions_path: str) -> np.ndarray:
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


def mean_prediction_error(predictions_path: str) -> np.ndarray:
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
